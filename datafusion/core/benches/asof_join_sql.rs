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

//! Criterion benchmark comparing an as-of join expressed as SQL across builds.
//!
//! The query
//!
//! ```sql
//! SELECT ... FROM left l JOIN right r ON l.sym = r.sym AND l.t >= r.t
//! ```
//!
//! is run through a full `SessionContext`. On a build that carries the AsOf
//! join optimizer rule (`try_asof_join` inside `JoinSelection`), the physical
//! plan is rewritten to `AsOfJoinExec`, which emits a single nearest match per
//! left row. On a build without the rule (e.g. `branch-53`), the planner falls
//! back to a hash join plus an inequality filter, which materializes *every*
//! right row with `r.t <= l.t`.
//!
//! The two strategies are therefore "similar SQL" rather than identical
//! semantics, but running this same binary on both branches gives an
//! apples-to-apples comparison of how each executes the query. On startup the
//! benchmark prints the physical plan so the active strategy is visible in the
//! output.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use tokio::runtime::Runtime;

/// Left/right table sizes and partition count. Kept modest because the
/// no-asof (hash join + filter) plan materializes an inequality join whose
/// output grows with the per-`sym` partition size.
const LEFT_ROWS: usize = 50_000;
const RIGHT_ROWS: usize = 50_000;
const NUM_SYMS: usize = 2_000;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Int64, false),
        Field::new("t", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn batches(num_rows: usize, num_syms: usize, schema: &SchemaRef) -> Vec<RecordBatch> {
    let syms: Vec<i64> = (0..num_rows).map(|i| (i % num_syms) as i64).collect();
    let ts: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    let vals: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(syms)),
            Arc::new(Int64Array::from(ts)),
            Arc::new(Int64Array::from(vals)),
        ],
    )
    .unwrap();

    let batch_size = 8192;
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < batch.num_rows() {
        let len = (batch.num_rows() - offset).min(batch_size);
        out.push(batch.slice(offset, len));
        offset += len;
    }
    out
}

fn make_ctx() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    let s = schema();
    let left = MemTable::try_new(Arc::clone(&s), vec![batches(LEFT_ROWS, NUM_SYMS, &s)])?;
    let right =
        MemTable::try_new(Arc::clone(&s), vec![batches(RIGHT_ROWS, NUM_SYMS, &s)])?;
    ctx.register_table("left_tbl", Arc::new(left))?;
    ctx.register_table("right_tbl", Arc::new(right))?;
    Ok(ctx)
}

/// Backward as-of: for each left row, the right row(s) with `r.t <= l.t`.
///
/// All six columns are selected so no projection is pushed into the join; this
/// keeps the comparison on the raw join/match cost and sidesteps the operator's
/// (currently broken) projection path.
const SQL_INNER: &str = "SELECT l.sym, l.t, l.v, r.sym AS r_sym, r.t AS r_t, r.v AS r_v \
     FROM left_tbl l JOIN right_tbl r ON l.sym = r.sym AND l.t >= r.t";
const SQL_LEFT: &str = "SELECT l.sym, l.t, l.v, r.sym AS r_sym, r.t AS r_t, r.v AS r_v \
     FROM left_tbl l LEFT JOIN right_tbl r ON l.sym = r.sym AND l.t >= r.t";

fn run_sql(ctx: &SessionContext, rt: &Runtime, sql: &str) {
    let df = rt.block_on(ctx.sql(sql)).unwrap();
    black_box(rt.block_on(df.collect()).unwrap());
}

/// Print the physical plan once so the output records which execution strategy
/// (AsOfJoinExec vs hash join + filter) this build selected.
fn report_plan(ctx: &SessionContext, rt: &Runtime, label: &str, sql: &str) {
    let plan = rt
        .block_on(async {
            let df = ctx.sql(sql).await?;
            df.create_physical_plan().await
        })
        .unwrap();
    let displayable =
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true);
    eprintln!("\n===== physical plan [{label}] =====\n{displayable}");
}

fn bench_asof_sql(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ctx = make_ctx().unwrap();

    report_plan(&ctx, &rt, "inner_backward", SQL_INNER);
    report_plan(&ctx, &rt, "left_backward", SQL_LEFT);

    let mut group = c.benchmark_group("asof_join_sql");

    group.bench_function("inner_backward", |b| {
        b.iter(|| run_sql(&ctx, &rt, SQL_INNER))
    });
    group.bench_function("left_backward", |b| {
        b.iter(|| run_sql(&ctx, &rt, SQL_LEFT))
    });

    group.finish();
}

criterion_group!(benches, bench_asof_sql);
criterion_main!(benches);
