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

//! nearest-match inequality join

use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::utils::{
    build_batch_from_indices, build_join_schema, check_join_is_valid,
    estimate_join_statistics, ColumnIndex,
};
use super::JoinOn;
use crate::common::can_project;
use crate::execution_plan::{boundedness_from_children, EmissionType};
use crate::joins::JoinOnRef;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::projection::{EmbeddedProjection, ProjectionExec};
use crate::sorts::sort::sort_batch;
use crate::{
    common, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, PhysicalSortExpr, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};

use arrow::array::{ArrayRef, UInt32Builder, UInt64Builder};
use arrow::compute::{concat_batches, SortOptions};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::{
    exec_err, internal_err, plan_err, project_schema, JoinSide, JoinType, NullEquality,
    Result,
};
use datafusion_execution::TaskContext;
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::equivalence::{
    join_equivalence_properties, ProjectionMapping,
};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_expr_common::sort_expr::LexOrdering;

use futures::future::{BoxFuture, FutureExt};
use futures::{ready, Stream};

/// whether an as-of join searches backward or forward from each left row
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOfJoinDirection {
    /// looks for the greatest right-side as-of key that is less than, or equal
    /// to, the left-side key
    Backward,
    /// looks for the smallest right-side as-of key that is greater than, or
    /// equal to, the left-side key
    Forward,
}

/// the inequality predicate that drives an as-of join
#[derive(Debug, Clone)]
pub struct AsOfJoinCondition {
    /// as-of key expression evaluated against the left input
    pub left: PhysicalExprRef,
    /// as-of key expression evaluated against the right input
    pub right: PhysicalExprRef,
    /// search direction derived from the inequality
    pub direction: AsOfJoinDirection,
    /// whether the inequality is strict
    pub strict: bool,
}

impl AsOfJoinCondition {
    /// creates an as-of condition from a normalized `left op right`
    /// inequality
    pub fn try_new(
        left: PhysicalExprRef,
        op: Operator,
        right: PhysicalExprRef,
    ) -> Result<Self> {
        let (direction, strict) = match op {
            Operator::Gt => (AsOfJoinDirection::Backward, true),
            Operator::GtEq => (AsOfJoinDirection::Backward, false),
            Operator::Lt => (AsOfJoinDirection::Forward, true),
            Operator::LtEq => (AsOfJoinDirection::Forward, false),
            _ => {
                return plan_err!(
                    "AsOfJoinCondition requires an inequality operator, got {op}"
                )
            }
        };

        Ok(Self {
            left,
            right,
            direction,
            strict,
        })
    }

    fn sort_options(&self) -> SortOptions {
        // for a forward asof join we flip the asof sort order so that lets the
        // same merge code keep the "last row that still passes" and still get
        // the nearest match instead of the farthest one
        let descending = matches!(self.direction, AsOfJoinDirection::Forward);
        SortOptions::new(descending, false)
    }
}

/// join execution plan that partitions rows by equality keys and emits at most
/// one nearest right-side row for each left-side row that satisfies the as-of
/// inequality
///
/// this initial implementation is a bounded, in-memory operator and it requests
/// single-partition inputs, collects both sides, sorts them by equality keys and
/// the as-of key, then does a linear merge inside each equality partition
#[derive(Debug, Clone)]
pub struct AsOfJoinExec {
    /// left input
    pub(crate) left: Arc<dyn ExecutionPlan>,
    /// right input
    pub(crate) right: Arc<dyn ExecutionPlan>,
    /// equality predicates used as partition keys
    pub(crate) on: JoinOn,
    /// inequality predicate used as the as-of key
    pub(crate) asof_condition: AsOfJoinCondition,
    /// how the join is performed
    pub(crate) join_type: JoinType,
    /// unprojected join schema
    join_schema: SchemaRef,
    /// information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// projection to apply to the output of the join
    pub projection: Option<Vec<usize>>,
    /// defines null equality for the equality partition keys
    pub(crate) null_equality: NullEquality,
    /// execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// cache holding plan properties
    cache: Arc<PlanProperties>,
}

impl AsOfJoinExec {
    /// tries to create a new as-of join
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        asof_condition: AsOfJoinCondition,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
        null_equality: NullEquality,
    ) -> Result<Self> {
        if !matches!(join_type, JoinType::Inner | JoinType::Left) {
            return plan_err!(
                "AsOfJoinExec supports Inner and Left joins, got {join_type:?}"
            );
        }

        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &on)?;

        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &join_type);
        let join_schema = Arc::new(join_schema);
        can_project(&join_schema, projection.as_deref())?;

        let cache = Arc::new(Self::compute_properties(
            &left,
            &right,
            Arc::clone(&join_schema),
            join_type,
            &on,
            projection.as_ref(),
        )?);

        Ok(Self {
            left,
            right,
            on,
            asof_condition,
            join_type,
            join_schema,
            column_indices,
            projection,
            null_equality,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// left input
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right input
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// equality predicates used as partition keys
    pub fn on(&self) -> JoinOnRef<'_> {
        &self.on
    }

    /// inequality predicate used as the as-of key
    pub fn asof_condition(&self) -> &AsOfJoinCondition {
        &self.asof_condition
    }

    /// how the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// defines null equality for equality partition keys
    pub fn null_equality(&self) -> NullEquality {
        self.null_equality
    }

    /// returns whether the join contains a projection
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// returns a new as-of join with the given projection
    pub fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        can_project(&self.schema(), projection.as_deref())?;
        let projection = match projection {
            Some(projection) => match &self.projection {
                Some(p) => Some(projection.iter().map(|i| p[*i]).collect()),
                None => Some(projection),
            },
            None => None,
        };
        Self::try_new(
            Arc::clone(&self.left),
            Arc::clone(&self.right),
            self.on.clone(),
            self.asof_condition.clone(),
            self.join_type,
            projection,
            self.null_equality,
        )
    }

    fn maintains_input_order(_join_type: JoinType) -> Vec<bool> {
        vec![false, false]
    }

    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        on: JoinOnRef<'_>,
        projection: Option<&Vec<usize>>,
    ) -> Result<PlanProperties> {
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            Arc::clone(&schema),
            &Self::maintains_input_order(join_type),
            None,
            on,
        )?;

        let mut output_partitioning =
            datafusion_physical_expr::Partitioning::UnknownPartitioning(1);

        if let Some(projection) = projection {
            let projection_mapping =
                ProjectionMapping::from_indices(projection, &schema)?;
            let out_schema = project_schema(&schema, Some(projection))?;
            output_partitioning =
                output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Final,
            boundedness_from_children([left, right]),
        ))
    }
}

impl DisplayAs for AsOfJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_on = self
                    .on
                    .iter()
                    .map(|(l, r)| {
                        format!(
                            "({l}, {r})",
                            l = fmt_sql(l.as_ref()),
                            r = fmt_sql(r.as_ref())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let op = match (self.asof_condition.direction, self.asof_condition.strict)
                {
                    (AsOfJoinDirection::Backward, true) => ">",
                    (AsOfJoinDirection::Backward, false) => ">=",
                    (AsOfJoinDirection::Forward, true) => "<",
                    (AsOfJoinDirection::Forward, false) => "<=",
                };
                let display_projection = if self.contains_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "".to_string()
                };
                write!(
                    f,
                    "AsOfJoinExec: join_type={:?}, on=[{}], asof={} {} {}{}",
                    self.join_type,
                    display_on,
                    fmt_sql(self.asof_condition.left.as_ref()),
                    op,
                    fmt_sql(self.asof_condition.right.as_ref()),
                    display_projection
                )
            }
            DisplayFormatType::TreeRender => {
                if self.join_type != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for AsOfJoinExec {
    fn name(&self) -> &'static str {
        "AsOfJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition, Distribution::SinglePartition]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.on.clone(),
            self.asof_condition.clone(),
            self.join_type,
            self.projection.clone(),
            self.null_equality,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "Invalid AsOfJoinExec partition {partition}; this operator produces one partition"
            );
        }

        let left = self.left.execute(0, Arc::clone(&context))?;
        let right = self.right.execute(0, Arc::clone(&context))?;

        let column_indices_after_projection = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        Ok(Box::pin(AsOfJoinStream::new(
            self.schema(),
            left,
            right,
            Arc::clone(&self.join_schema),
            column_indices_after_projection,
            self.on.clone(),
            self.asof_condition.clone(),
            self.join_type,
            self.null_equality,
            context.session_config().batch_size(),
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        if partition.is_some() {
            return Ok(Statistics::new_unknown(&self.schema()));
        }
        estimate_join_statistics(
            self.left.partition_statistics(None)?,
            self.right.partition_statistics(None)?,
            &self.on,
            &self.join_type,
            &self.join_schema,
        )
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if self.contains_projection() {
            return Ok(None);
        }

        crate::projection::try_embed_projection(projection, self)
    }
}

impl EmbeddedProjection for AsOfJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.with_projection(projection)
    }
}

/// stream returned by the as-of join
pub struct AsOfJoinStream {
    schema: SchemaRef,
    state: AsOfJoinStreamState,
    batch_size: usize,
    baseline_metrics: BaselineMetrics,
}

enum AsOfJoinStreamState {
    Loading(BoxFuture<'static, Result<RecordBatch>>),
    Draining { batch: RecordBatch, offset: usize },
    Done,
}

impl AsOfJoinStream {
    #[allow(clippy::too_many_arguments)]
    fn new(
        schema: SchemaRef,
        left: SendableRecordBatchStream,
        right: SendableRecordBatchStream,
        join_schema: SchemaRef,
        column_indices: Vec<ColumnIndex>,
        on: JoinOn,
        asof_condition: AsOfJoinCondition,
        join_type: JoinType,
        null_equality: NullEquality,
        batch_size: usize,
        baseline_metrics: BaselineMetrics,
    ) -> Self {
        let state = AsOfJoinStreamState::Loading(
            async move {
                let left_schema = left.schema();
                let right_schema = right.schema();
                let left = concat_or_empty(left_schema, common::collect(left).await?)?;
                let right = concat_or_empty(right_schema, common::collect(right).await?)?;

                join_asof(
                    left,
                    right,
                    join_schema,
                    column_indices,
                    on,
                    asof_condition,
                    join_type,
                    null_equality,
                )
            }
            .boxed(),
        );

        Self {
            schema,
            state,
            batch_size,
            baseline_metrics,
        }
    }
}

impl Stream for AsOfJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            match &mut this.state {
                AsOfJoinStreamState::Loading(fut) => {
                    match ready!(fut.as_mut().poll(cx)) {
                        Ok(batch) if batch.num_rows() == 0 => {
                            this.state = AsOfJoinStreamState::Done;
                            return Poll::Ready(None);
                        }
                        Ok(batch) => {
                            this.state =
                                AsOfJoinStreamState::Draining { batch, offset: 0 };
                        }
                        Err(e) => {
                            this.state = AsOfJoinStreamState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
                AsOfJoinStreamState::Draining { batch, offset } => {
                    if *offset >= batch.num_rows() {
                        this.state = AsOfJoinStreamState::Done;
                        return Poll::Ready(None);
                    }
                    let len = (batch.num_rows() - *offset).min(this.batch_size);
                    let output = batch.slice(*offset, len);
                    *offset += len;
                    this.baseline_metrics.record_output(len);
                    return Poll::Ready(Some(Ok(output)));
                }
                AsOfJoinStreamState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for AsOfJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn concat_or_empty(schema: SchemaRef, batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    if batches.is_empty() {
        Ok(RecordBatch::new_empty(schema))
    } else {
        Ok(concat_batches(&schema, &batches)?)
    }
}

#[allow(clippy::too_many_arguments)]
fn join_asof(
    left: RecordBatch,
    right: RecordBatch,
    schema: SchemaRef,
    column_indices: Vec<ColumnIndex>,
    on: JoinOn,
    asof_condition: AsOfJoinCondition,
    join_type: JoinType,
    null_equality: NullEquality,
) -> Result<RecordBatch> {
    if right.num_rows() > u32::MAX as usize {
        return exec_err!(
            "AsOfJoinExec does not support right inputs with more than {} rows",
            u32::MAX
        );
    }

    let left = sort_join_input(&left, &on, &asof_condition, JoinSide::Left)?;
    let right = sort_join_input(&right, &on, &asof_condition, JoinSide::Right)?;

    let left_keys = SortedKeyValues::try_new(
        &left,
        &on,
        &asof_condition,
        JoinSide::Left,
        null_equality,
    )?;
    let right_keys = SortedKeyValues::try_new(
        &right,
        &on,
        &asof_condition,
        JoinSide::Right,
        null_equality,
    )?;

    let mut left_indices = UInt64Builder::new();
    let mut right_indices = UInt32Builder::new();

    let mut left_pos = 0;
    let mut right_pos = 0;

    // both sides are sorted by the equality keys first, so the outer loop can
    // move partition by partition, when a left partition has no matching right
    // partition, left joins emit nulls and inner joins simply skip it
    while left_pos < left.num_rows() {
        if right_pos >= right.num_rows() {
            append_unmatched_left_partition(
                left_pos,
                left.num_rows(),
                join_type,
                &mut left_indices,
                &mut right_indices,
            );
            break;
        }

        match left_keys.partition_cmp(left_pos, &right_keys, right_pos) {
            Ordering::Less => {
                let left_end = left_keys.partition_end(left_pos);
                append_unmatched_left_partition(
                    left_pos,
                    left_end,
                    join_type,
                    &mut left_indices,
                    &mut right_indices,
                );
                left_pos = left_end;
            }
            Ordering::Greater => {
                right_pos = right_keys.partition_end(right_pos);
            }
            Ordering::Equal => {
                let left_end = left_keys.partition_end(left_pos);
                let right_end = right_keys.partition_end(right_pos);
                append_asof_partition(
                    &left_keys,
                    &right_keys,
                    left_pos,
                    left_end,
                    right_pos,
                    right_end,
                    join_type,
                    asof_condition.strict,
                    &mut left_indices,
                    &mut right_indices,
                );
                left_pos = left_end;
                right_pos = right_end;
            }
        }
    }

    build_batch_from_indices(
        &schema,
        &left,
        &right,
        &left_indices.finish(),
        &right_indices.finish(),
        &column_indices,
        JoinSide::Left,
        join_type,
    )
}

fn append_unmatched_left_partition(
    left_start: usize,
    left_end: usize,
    join_type: JoinType,
    left_indices: &mut UInt64Builder,
    right_indices: &mut UInt32Builder,
) {
    if join_type != JoinType::Left {
        return;
    }

    for left_idx in left_start..left_end {
        left_indices.append_value(left_idx as u64);
        right_indices.append_null();
    }
}

#[allow(clippy::too_many_arguments)]
fn append_asof_partition(
    left_keys: &SortedKeyValues,
    right_keys: &SortedKeyValues,
    left_start: usize,
    left_end: usize,
    right_start: usize,
    right_end: usize,
    join_type: JoinType,
    strict: bool,
    left_indices: &mut UInt64Builder,
    right_indices: &mut UInt32Builder,
) {
    let mut right_pos = right_start;
    let mut last_match = None;

    // within one equality partition, the asof keys are sorted in the direction
    // that makes a single right-side cursor enough and every right row we pass is
    // a better candidate than the previous one, so `last_match` is the nearest
    // right row seen so far for the current and later left rows
    for left_idx in left_start..left_end {
        if !left_keys.is_joinable(left_idx) {
            append_unmatched_left(left_idx, join_type, left_indices, right_indices);
            continue;
        }

        while right_pos < right_end {
            if !right_keys.is_joinable(right_pos) {
                right_pos += 1;
                continue;
            }

            if right_keys.passes_asof(right_pos, left_keys, left_idx, strict) {
                last_match = Some(right_pos);
                right_pos += 1;
            } else {
                break;
            }
        }

        if let Some(right_idx) = last_match {
            left_indices.append_value(left_idx as u64);
            right_indices.append_value(right_idx as u32);
        } else {
            append_unmatched_left(left_idx, join_type, left_indices, right_indices);
        }
    }
}

fn append_unmatched_left(
    left_idx: usize,
    join_type: JoinType,
    left_indices: &mut UInt64Builder,
    right_indices: &mut UInt32Builder,
) {
    if join_type == JoinType::Left {
        left_indices.append_value(left_idx as u64);
        right_indices.append_null();
    }
}

fn sort_join_input(
    batch: &RecordBatch,
    on: JoinOnRef<'_>,
    asof_condition: &AsOfJoinCondition,
    side: JoinSide,
) -> Result<RecordBatch> {
    let mut sort_exprs = on
        .iter()
        .map(|(left, right)| PhysicalSortExpr {
            expr: Arc::clone(if side == JoinSide::Left { left } else { right }),
            options: SortOptions::new(false, false),
        })
        .collect::<Vec<_>>();

    sort_exprs.push(PhysicalSortExpr {
        expr: Arc::clone(if side == JoinSide::Left {
            &asof_condition.left
        } else {
            &asof_condition.right
        }),
        options: asof_condition.sort_options(),
    });

    let Some(sort_exprs) = LexOrdering::new(sort_exprs) else {
        return plan_err!("AsOfJoinExec requires at least one as-of sort expression");
    };

    sort_batch(batch, &sort_exprs, None)
}

struct SortedKeyValues {
    partition_rows: Option<Rows>,
    asof_rows: Rows,
    partition_valid: Vec<bool>,
    asof_valid: Vec<bool>,
    row_count: usize,
}

impl SortedKeyValues {
    fn try_new(
        batch: &RecordBatch,
        on: JoinOnRef<'_>,
        asof_condition: &AsOfJoinCondition,
        side: JoinSide,
        null_equality: NullEquality,
    ) -> Result<Self> {
        let partition_arrays = on
            .iter()
            .map(|(left, right)| {
                evaluate_expr(if side == JoinSide::Left { left } else { right }, batch)
            })
            .collect::<Result<Vec<_>>>()?;
        let asof_array = evaluate_expr(
            if side == JoinSide::Left {
                &asof_condition.left
            } else {
                &asof_condition.right
            },
            batch,
        )?;

        let partition_valid =
            build_partition_validity(&partition_arrays, batch.num_rows(), null_equality);
        let asof_valid = build_validity(&asof_array, batch.num_rows());
        let partition_rows = if partition_arrays.is_empty() {
            None
        } else {
            Some(convert_to_rows(
                &partition_arrays,
                SortOptions::new(false, false),
            )?)
        };
        let asof_rows = convert_to_rows(&[asof_array], asof_condition.sort_options())?;

        Ok(Self {
            partition_rows,
            asof_rows,
            partition_valid,
            asof_valid,
            row_count: batch.num_rows(),
        })
    }

    fn partition_cmp(
        &self,
        left_idx: usize,
        right: &SortedKeyValues,
        right_idx: usize,
    ) -> Ordering {
        match (&self.partition_rows, &right.partition_rows) {
            (Some(left_rows), Some(right_rows)) => {
                left_rows.row(left_idx).cmp(&right_rows.row(right_idx))
            }
            (None, None) => Ordering::Equal,
            _ => unreachable!("AsOfJoinExec left/right partition key counts differ"),
        }
    }

    fn partition_end(&self, start: usize) -> usize {
        let Some(partition_rows) = &self.partition_rows else {
            return self.row_count;
        };

        let mut end = start + 1;
        while end < self.row_count && partition_rows.row(start) == partition_rows.row(end)
        {
            end += 1;
        }
        end
    }

    fn is_joinable(&self, row: usize) -> bool {
        self.partition_valid[row] && self.asof_valid[row]
    }

    fn passes_asof(
        &self,
        row: usize,
        left: &SortedKeyValues,
        left_row: usize,
        strict: bool,
    ) -> bool {
        // `self` is the right side here because forward joins sort the asof key
        // descending, the comparison still reads as "has this right row moved
        // far enough toward the left row to be a valid candidate?"
        let ord = self.asof_rows.row(row).cmp(&left.asof_rows.row(left_row));
        if strict {
            ord == Ordering::Less
        } else {
            ord != Ordering::Greater
        }
    }
}

fn evaluate_expr(expr: &PhysicalExprRef, batch: &RecordBatch) -> Result<ArrayRef> {
    match expr.evaluate(batch)? {
        ColumnarValue::Array(array) => Ok(array),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(batch.num_rows()),
    }
}

fn convert_to_rows(arrays: &[ArrayRef], options: SortOptions) -> Result<Rows> {
    let sort_fields = arrays
        .iter()
        .map(|array| SortField::new_with_options(array.data_type().clone(), options))
        .collect();
    Ok(RowConverter::new(sort_fields)?.convert_columns(arrays)?)
}

fn build_partition_validity(
    arrays: &[ArrayRef],
    row_count: usize,
    null_equality: NullEquality,
) -> Vec<bool> {
    if null_equality == NullEquality::NullEqualsNull {
        return vec![true; row_count];
    }

    let mut valid = vec![true; row_count];
    for array in arrays {
        if array.null_count() == 0 {
            continue;
        }
        for (row, is_valid) in valid.iter_mut().enumerate() {
            *is_valid &= !array.is_null(row);
        }
    }
    valid
}

fn build_validity(array: &ArrayRef, row_count: usize) -> Vec<bool> {
    if array.null_count() == 0 {
        return vec![true; row_count];
    }

    (0..row_count).map(|row| !array.is_null(row)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::assert_batches_eq;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::expressions::Column;

    use crate::test::TestMemoryExec;

    fn col(name: &str, index: usize) -> PhysicalExprRef {
        Arc::new(Column::new(name, index))
    }

    fn table(
        names: [&str; 3],
        cols: [Vec<Option<i32>>; 3],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(names[0], DataType::Int32, true),
            Field::new(names[1], DataType::Int32, true),
            Field::new(names[2], DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(cols[0].clone())),
                Arc::new(Int32Array::from(cols[1].clone())),
                Arc::new(Int32Array::from(cols[2].clone())),
            ],
        )?;

        Ok(TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?)
    }

    async fn collect_join(join: AsOfJoinExec) -> Result<Vec<RecordBatch>> {
        let stream = join.execute(0, Arc::new(TaskContext::default()))?;
        common::collect(stream).await
    }

    #[tokio::test]
    async fn backward_left_asof_join_partitions_and_null_extends() -> Result<()> {
        let left = table(
            ["l_sym", "l_t", "l_v"],
            [
                vec![Some(1), Some(1), Some(2), Some(2), Some(3)],
                vec![Some(10), Some(12), Some(5), Some(2), Some(7)],
                vec![Some(100), Some(120), Some(200), Some(220), Some(300)],
            ],
        )?;
        let right = table(
            ["r_sym", "r_t", "r_v"],
            [
                vec![Some(1), Some(1), Some(2), Some(2), Some(4)],
                vec![Some(9), Some(11), Some(3), Some(6), Some(1)],
                vec![Some(90), Some(110), Some(230), Some(260), Some(400)],
            ],
        )?;

        let join = AsOfJoinExec::try_new(
            left,
            right,
            vec![(col("l_sym", 0), col("r_sym", 0))],
            AsOfJoinCondition::try_new(col("l_t", 1), Operator::GtEq, col("r_t", 1))?,
            JoinType::Left,
            None,
            NullEquality::NullEqualsNothing,
        )?;

        let batches = collect_join(join).await?;
        let expected = [
            "+-------+-----+-----+-------+-----+-----+",
            "| l_sym | l_t | l_v | r_sym | r_t | r_v |",
            "+-------+-----+-----+-------+-----+-----+",
            "| 1     | 10  | 100 | 1     | 9   | 90  |",
            "| 1     | 12  | 120 | 1     | 11  | 110 |",
            "| 2     | 2   | 220 |       |     |     |",
            "| 2     | 5   | 200 | 2     | 3   | 230 |",
            "| 3     | 7   | 300 |       |     |     |",
            "+-------+-----+-----+-------+-----+-----+",
        ];
        assert_batches_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn forward_inner_asof_join_flips_sort_direction() -> Result<()> {
        let left = table(
            ["l_sym", "l_t", "l_v"],
            [
                vec![Some(1), Some(1), Some(1)],
                vec![Some(10), Some(12), Some(15)],
                vec![Some(100), Some(120), Some(150)],
            ],
        )?;
        let right = table(
            ["r_sym", "r_t", "r_v"],
            [
                vec![Some(1), Some(1), Some(1)],
                vec![Some(9), Some(11), Some(20)],
                vec![Some(90), Some(110), Some(200)],
            ],
        )?;

        let join = AsOfJoinExec::try_new(
            left,
            right,
            vec![(col("l_sym", 0), col("r_sym", 0))],
            AsOfJoinCondition::try_new(col("l_t", 1), Operator::LtEq, col("r_t", 1))?,
            JoinType::Inner,
            None,
            NullEquality::NullEqualsNothing,
        )?;

        let batches = collect_join(join).await?;
        let expected = [
            "+-------+-----+-----+-------+-----+-----+",
            "| l_sym | l_t | l_v | r_sym | r_t | r_v |",
            "+-------+-----+-----+-------+-----+-----+",
            "| 1     | 15  | 150 | 1     | 20  | 200 |",
            "| 1     | 12  | 120 | 1     | 20  | 200 |",
            "| 1     | 10  | 100 | 1     | 11  | 110 |",
            "+-------+-----+-----+-------+-----+-----+",
        ];
        assert_batches_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn null_keys_do_not_match_for_default_null_equality() -> Result<()> {
        let left = table(
            ["l_sym", "l_t", "l_v"],
            [
                vec![None, Some(1), Some(1)],
                vec![Some(5), None, Some(10)],
                vec![Some(50), Some(100), Some(110)],
            ],
        )?;
        let right = table(
            ["r_sym", "r_t", "r_v"],
            [
                vec![None, Some(1), Some(1)],
                vec![Some(4), Some(9), None],
                vec![Some(40), Some(90), Some(999)],
            ],
        )?;

        let join = AsOfJoinExec::try_new(
            left,
            right,
            vec![(col("l_sym", 0), col("r_sym", 0))],
            AsOfJoinCondition::try_new(col("l_t", 1), Operator::GtEq, col("r_t", 1))?,
            JoinType::Left,
            None,
            NullEquality::NullEqualsNothing,
        )?;

        let batches = collect_join(join).await?;
        let expected = [
            "+-------+-----+-----+-------+-----+-----+",
            "| l_sym | l_t | l_v | r_sym | r_t | r_v |",
            "+-------+-----+-----+-------+-----+-----+",
            "| 1     | 10  | 110 | 1     | 9   | 90  |",
            "| 1     |     | 100 |       |     |     |",
            "|       | 5   | 50  |       |     |     |",
            "+-------+-----+-----+-------+-----+-----+",
        ];
        assert_batches_eq!(expected, &batches);

        Ok(())
    }
}
