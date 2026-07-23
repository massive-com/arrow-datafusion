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

//! Criterion benchmarks for AsOf Join
//!
//! These benchmarks measure `AsOfJoinExec` end-to-end: each side is fed as
//! in-memory `RecordBatch`es and the operator collects, sorts by the equality
//! keys and the time column, and performs the as-of match. The internal sort
//! is therefore included in the measurement, which reflects how the operator
//! behaves in a plan.
//!
//! Data model: `(sym: Int64, t: Int64, v: Int64)` where `sym` is the equality
//! ("partition") key and `t` is the ordering / time column. `num_syms`
//! controls how many distinct partitions the rows are spread across, which
//! drives the partition sizes the match loop walks over.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::collect;
use datafusion_physical_plan::joins::utils::JoinOn;
use datafusion_physical_plan::joins::{AsOfJoinCondition, AsOfJoinExec};
use datafusion_physical_plan::test::TestMemoryExec;
use tokio::runtime::Runtime;

/// Build in-memory batches (split into ~8192-row chunks).
///
/// Schema: `(sym: Int64, t: Int64, v: Int64)`.
///
/// `sym = row_index % num_syms` and `t = row_index`, so each `sym` partition
/// carries a strictly increasing sequence of timestamps once sorted.
///
/// The operator always sorts its inputs internally. When `sorted` is true the
/// rows are pre-sorted by `(sym, t)`, so that internal sort sees already-ordered
/// input and is effectively free — isolating the join/match kernel. When false
/// the rows are in `t` order (syms interleaved), so the internal sort does real
/// work and is included in the measurement.
fn build_batches(
    num_rows: usize,
    num_syms: usize,
    sorted: bool,
    schema: &SchemaRef,
) -> Vec<RecordBatch> {
    // Row order: by default `t == row_index` ascending with syms interleaved.
    // When `sorted`, stably reorder by sym so rows come out in (sym, t) order.
    let mut order: Vec<usize> = (0..num_rows).collect();
    if sorted {
        order.sort_by_key(|&i| (i % num_syms) as i64);
    }
    let syms: Vec<i64> = order.iter().map(|&i| (i % num_syms) as i64).collect();
    let ts: Vec<i64> = order.iter().map(|&i| i as i64).collect();
    let vals: Vec<i64> = order.iter().map(|&i| i as i64).collect();

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
    let mut batches = Vec::new();
    let mut offset = 0;
    while offset < batch.num_rows() {
        let len = (batch.num_rows() - offset).min(batch_size);
        batches.push(batch.slice(offset, len));
        offset += len;
    }
    batches
}

fn make_exec(batches: &[RecordBatch], schema: &SchemaRef) -> Arc<dyn ExecutionPlan> {
    TestMemoryExec::try_new_exec(&[batches.to_vec()], Arc::clone(schema), None).unwrap()
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Int64, false),
        Field::new("t", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// Run one as-of join to completion and return the produced row count.
///
/// `op` selects the as-of direction: `GtEq`/`Gt` are backward (nearest prior
/// right row), `LtEq`/`Lt` are forward (nearest following right row).
fn do_asof_join(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    op: Operator,
    join_type: JoinType,
    rt: &Runtime,
) -> usize {
    let on: JoinOn = vec![(
        col("sym", &left.schema()).unwrap(),
        col("sym", &right.schema()).unwrap(),
    )];
    let condition = AsOfJoinCondition::try_new(
        col("t", &left.schema()).unwrap(),
        op,
        col("t", &right.schema()).unwrap(),
    )
    .unwrap();
    let join = AsOfJoinExec::try_new(
        left,
        right,
        on,
        condition,
        join_type,
        None,
        NullEquality::NullEqualsNothing,
    )
    .unwrap();

    let task_ctx = Arc::new(TaskContext::default());
    rt.block_on(async {
        let batches = collect(Arc::new(join), task_ctx).await.unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    })
}

/// A single benchmark configuration.
struct Case {
    name: &'static str,
    left_rows: usize,
    right_rows: usize,
    num_syms: usize,
    op: Operator,
    join_type: JoinType,
}

/// Register one case: build both sides once, then re-create the leaf execs
/// each iteration (cheap `Arc` clones of the shared batches).
///
/// `sorted` selects whether the inputs are pre-sorted by `(sym, t)`: the
/// `presorted` variant makes the operator's internal sort a no-op so the
/// measurement isolates the join/match kernel; `sort_included` measures the
/// operator as it runs on unordered input.
fn run_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &Runtime,
    s: &SchemaRef,
    case: &Case,
    sorted: bool,
) {
    let left_batches = build_batches(case.left_rows, case.num_syms, sorted, s);
    let right_batches = build_batches(case.right_rows, case.num_syms, sorted, s);
    let variant = if sorted { "presorted" } else { "sort_included" };
    let id = BenchmarkId::new(format!("{}/{variant}", case.name), case.left_rows);
    group.bench_function(id, |b| {
        b.iter(|| {
            let left = make_exec(&left_batches, s);
            let right = make_exec(&right_batches, s);
            do_asof_join(left, right, case.op, case.join_type, rt)
        })
    });
}

fn bench_asof(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let s = schema();
    let n = 100_000;

    let cases = [
        // Backward Inner — nearest prior right row per (sym), 1K partitions.
        Case {
            name: "inner_backward",
            left_rows: n,
            right_rows: n,
            num_syms: 1_000,
            op: Operator::GtEq,
            join_type: JoinType::Inner,
        },
        // Backward Left — like above but unmatched left rows are null-extended.
        Case {
            name: "left_backward",
            left_rows: n,
            right_rows: n,
            num_syms: 1_000,
            op: Operator::GtEq,
            join_type: JoinType::Left,
        },
        // Backward Left, strict inequality (Gt) — excludes equal-timestamp matches.
        Case {
            name: "left_backward_strict",
            left_rows: n,
            right_rows: n,
            num_syms: 1_000,
            op: Operator::Gt,
            join_type: JoinType::Left,
        },
        // Forward Inner — nearest following right row (flips the sort direction).
        Case {
            name: "inner_forward",
            left_rows: n,
            right_rows: n,
            num_syms: 1_000,
            op: Operator::LtEq,
            join_type: JoinType::Inner,
        },
        // Few partitions — 10 syms, so the match loop walks long per-sym runs.
        Case {
            name: "left_backward_few_partitions",
            left_rows: n,
            right_rows: n,
            num_syms: 10,
            op: Operator::GtEq,
            join_type: JoinType::Left,
        },
        // Many partitions — 50K syms, so partitions are tiny (~2 rows each).
        Case {
            name: "left_backward_many_partitions",
            left_rows: n,
            right_rows: n,
            num_syms: 50_000,
            op: Operator::GtEq,
            join_type: JoinType::Left,
        },
        // Asymmetric — large left probed against a small right side.
        Case {
            name: "left_backward_asymmetric",
            left_rows: n,
            right_rows: n / 10,
            num_syms: 1_000,
            op: Operator::GtEq,
            join_type: JoinType::Left,
        },
    ];

    let mut group = c.benchmark_group("asof_join");

    // Every case measured on unordered input (internal sort included).
    for case in &cases {
        run_case(&mut group, &rt, &s, case, false);
    }

    // Pre-sorted counterparts for the two primary cases, to isolate the
    // join/match kernel from the internal sort cost.
    for case in cases
        .iter()
        .filter(|c| matches!(c.name, "inner_backward" | "left_backward"))
    {
        run_case(&mut group, &rt, &s, case, true);
    }

    group.finish();
}

criterion_group!(benches, bench_asof);
criterion_main!(benches);
