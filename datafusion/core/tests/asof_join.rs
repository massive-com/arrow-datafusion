// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{Result, assert_batches_eq, assert_contains};

/// trades(symbol, t, price): rows the quotes align against.
fn trades_batch() -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("price", DataType::Int64, false),
    ]));
    let symbol = Arc::new(StringArray::from(vec!["AAPL", "AAPL", "MSFT"]));
    let t = Arc::new(Int64Array::from(vec![10, 20, 10]));
    let price = Arc::new(Int64Array::from(vec![100, 200, 50]));
    RecordBatch::try_new(schema, vec![symbol, t, price]).map_err(Into::into)
}

/// quotes(symbol, t, bid): the left/driving side of the ASOF join.
fn quotes_batch() -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("bid", DataType::Int64, false),
    ]));
    let symbol = Arc::new(StringArray::from(vec!["AAPL", "AAPL", "AAPL", "MSFT"]));
    let t = Arc::new(Int64Array::from(vec![5, 15, 25, 5]));
    let bid = Arc::new(Int64Array::from(vec![1, 2, 3, 9]));
    RecordBatch::try_new(schema, vec![symbol, t, bid]).map_err(Into::into)
}

const ASOF_QUERY: &str = "SELECT q.symbol, q.t, tr.t AS trade_t, tr.price \
     FROM quotes q \
     ASOF JOIN trades tr \
     MATCH_CONDITION (q.t >= tr.t) \
     ON q.symbol = tr.symbol \
     ORDER BY q.symbol, q.t";

/// For each quote, the join must return the single nearest earlier trade of the
/// same symbol (largest `tr.t` with `tr.t <= q.t`), and NULLs when none exists
/// (LEFT-join semantics: every quote row survives).
#[tokio::test]
async fn asof_join_nearest_earlier_trade() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_batch("trades", trades_batch()?)?;
    ctx.register_batch("quotes", quotes_batch()?)?;

    let results = ctx.sql(ASOF_QUERY).await?.collect().await?;

    assert_batches_eq!(
        &[
            "+--------+----+---------+-------+",
            "| symbol | t  | trade_t | price |",
            "+--------+----+---------+-------+",
            "| AAPL   | 5  |         |       |",
            "| AAPL   | 15 | 10      | 100   |",
            "| AAPL   | 25 | 20      | 200   |",
            "| MSFT   | 5  |         |       |",
            "+--------+----+---------+-------+",
        ],
        &results
    );
    Ok(())
}

/// The query must lower to the dedicated physical operator, not a rewrite into
/// a regular join + filter.
#[tokio::test]
async fn asof_join_uses_physical_operator() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_batch("trades", trades_batch()?)?;
    ctx.register_batch("quotes", quotes_batch()?)?;

    let physical_plan = ctx.sql(ASOF_QUERY).await?.create_physical_plan().await?;
    let displayed = displayable(physical_plan.as_ref()).indent(true).to_string();

    assert_contains!(displayed, "AsOfJoinExec");
    Ok(())
}

/// With multi-partition inputs and `target_partitions > 1`, the planner must
/// hash-partition both inputs on the equality keys (`RepartitionExec`) so that
/// same-key rows split across input partitions are brought together; each
/// partition is then joined independently and the results are unchanged.
#[tokio::test]
async fn asof_join_runs_partitioned_by_equality_keys() -> Result<()> {
    use datafusion::datasource::MemTable;

    fn batch(
        schema: &Arc<Schema>,
        symbols: Vec<&str>,
        ts: Vec<i64>,
        vals: Vec<i64>,
    ) -> Result<RecordBatch> {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(StringArray::from(symbols)),
                Arc::new(Int64Array::from(ts)),
                Arc::new(Int64Array::from(vals)),
            ],
        )
        .map_err(Into::into)
    }

    let trades_schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("price", DataType::Int64, false),
    ]));
    let quotes_schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("t", DataType::Int64, false),
        Field::new("bid", DataType::Int64, false),
    ]));

    // Same rows as the single-partition test, but split so AAPL straddles two
    // input partitions on each side (exercises cross-partition co-location).
    let trades = MemTable::try_new(
        Arc::clone(&trades_schema),
        vec![
            vec![batch(&trades_schema, vec!["AAPL"], vec![10], vec![100])?],
            vec![batch(
                &trades_schema,
                vec!["AAPL", "MSFT"],
                vec![20, 10],
                vec![200, 50],
            )?],
        ],
    )?;
    let quotes = MemTable::try_new(
        Arc::clone(&quotes_schema),
        vec![
            vec![batch(
                &quotes_schema,
                vec!["AAPL", "AAPL"],
                vec![5, 25],
                vec![1, 3],
            )?],
            vec![batch(
                &quotes_schema,
                vec!["AAPL", "MSFT"],
                vec![15, 5],
                vec![2, 9],
            )?],
        ],
    )?;

    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("trades", Arc::new(trades))?;
    ctx.register_table("quotes", Arc::new(quotes))?;

    // The plan must hash-partition both inputs on the equality keys.
    let physical_plan = ctx.sql(ASOF_QUERY).await?.create_physical_plan().await?;
    let displayed = displayable(physical_plan.as_ref()).indent(true).to_string();
    assert_contains!(displayed.clone(), "AsOfJoinExec");
    assert_contains!(displayed, "RepartitionExec: partitioning=Hash");

    // Results must be identical to single-partition execution.
    let results = ctx.sql(ASOF_QUERY).await?.collect().await?;
    assert_batches_eq!(
        &[
            "+--------+----+---------+-------+",
            "| symbol | t  | trade_t | price |",
            "+--------+----+---------+-------+",
            "| AAPL   | 5  |         |       |",
            "| AAPL   | 15 | 10      | 100   |",
            "| AAPL   | 25 | 20      | 200   |",
            "| MSFT   | 5  |         |       |",
            "+--------+----+---------+-------+",
        ],
        &results
    );
    Ok(())
}

/// With the ordering-aware path disabled via config, the operator sorts its
/// inputs internally (no required input ordering) and still returns the
/// correct nearest-match results.
#[tokio::test]
async fn asof_join_sorts_internally_when_disabled() -> Result<()> {
    let mut config = SessionConfig::new();
    config.options_mut().optimizer.asof_join_use_sorted_input = false;
    let ctx = SessionContext::new_with_config(config);
    ctx.register_batch("trades", trades_batch()?)?;
    ctx.register_batch("quotes", quotes_batch()?)?;

    let results = ctx.sql(ASOF_QUERY).await?.collect().await?;
    assert_batches_eq!(
        &[
            "+--------+----+---------+-------+",
            "| symbol | t  | trade_t | price |",
            "+--------+----+---------+-------+",
            "| AAPL   | 5  |         |       |",
            "| AAPL   | 15 | 10      | 100   |",
            "| AAPL   | 25 | 20      | 200   |",
            "| MSFT   | 5  |         |       |",
            "+--------+----+---------+-------+",
        ],
        &results
    );
    Ok(())
}
