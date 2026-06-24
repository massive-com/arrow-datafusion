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

//! The [`JoinSelection`] rule tries to modify a given plan so that it can
//! accommodate infinite sources and utilize statistical information (if there
//! is any) to obtain more performant plans. To achieve the first goal, it
//! tries to transform a non-runnable query (with the given infinite sources)
//! into a runnable query by replacing pipeline-breaking join operations with
//! pipeline-friendly ones. To achieve the second goal, it selects the proper
//! `PartitionMode` and the build side using the available statistics for hash joins.

use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{internal_err, JoinSide, JoinType};
use datafusion_expr::Operator;
use datafusion_expr_common::sort_properties::SortProperties;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::execution_plan::EmissionType;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter, JoinOn};
use datafusion_physical_plan::joins::{
    AsOfJoinCondition, AsOfJoinExec, CrossJoinExec, HashJoinExec, NestedLoopJoinExec,
    PartitionMode, StreamJoinPartitionMode, SymmetricHashJoinExec,
};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use std::sync::Arc;

/// The [`JoinSelection`] rule tries to modify a given plan so that it can
/// accommodate infinite sources and optimize joins in the plan according to
/// available statistical information, if there is any.
#[derive(Default, Debug)]
pub struct JoinSelection {}

impl JoinSelection {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

// TODO: We need some performance test for Right Semi/Right Join swap to Left Semi/Left Join in case that the right side is smaller but not much smaller.
// TODO: In PrestoSQL, the optimizer flips join sides only if one side is much smaller than the other by more than SIZE_DIFFERENCE_THRESHOLD times, by default is 8 times.
/// Checks statistics for join swap.
pub(crate) fn should_swap_join_order(
    left: &dyn ExecutionPlan,
    right: &dyn ExecutionPlan,
) -> Result<bool> {
    // Get the left and right table's total bytes
    // If both the left and right tables contain total_byte_size statistics,
    // use `total_byte_size` to determine `should_swap_join_order`, else use `num_rows`
    let left_stats = left.partition_statistics(None)?;
    let right_stats = right.partition_statistics(None)?;
    // First compare `total_byte_size` of left and right side,
    // if information in this field is insufficient fallback to the `num_rows`
    match (
        left_stats.total_byte_size.get_value(),
        right_stats.total_byte_size.get_value(),
    ) {
        (Some(l), Some(r)) => Ok(l > r),
        _ => match (
            left_stats.num_rows.get_value(),
            right_stats.num_rows.get_value(),
        ) {
            (Some(l), Some(r)) => Ok(l > r),
            _ => Ok(false),
        },
    }
}

fn supports_collect_by_thresholds(
    plan: &dyn ExecutionPlan,
    threshold_byte_size: usize,
    threshold_num_rows: usize,
) -> bool {
    // Currently we do not trust the 0 value from stats, due to stats collection might have bug
    // TODO check the logic in datasource::get_statistics_with_limit()
    let Ok(stats) = plan.partition_statistics(None) else {
        return false;
    };

    if let Some(byte_size) = stats.total_byte_size.get_value() {
        *byte_size != 0 && *byte_size < threshold_byte_size
    } else if let Some(num_rows) = stats.num_rows.get_value() {
        *num_rows != 0 && *num_rows < threshold_num_rows
    } else {
        false
    }
}

impl PhysicalOptimizerRule for JoinSelection {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // First, we make pipeline-fixing modifications to joins so as to accommodate
        // unbounded inputs. Each pipeline-fixing subrule, which is a function
        // of type `PipelineFixerSubrule`, takes a single [`PipelineStatePropagator`]
        // argument storing state variables that indicate the unboundedness status
        // of the current [`ExecutionPlan`] as we traverse the plan tree.
        let subrules: Vec<Box<PipelineFixerSubrule>> = vec![
            Box::new(hash_join_convert_symmetric_subrule),
            Box::new(hash_join_swap_subrule),
        ];
        let new_plan = plan
            .transform_up(|p| apply_subrules(p, &subrules, config))
            .data()?;
        // Next, we apply another subrule that tries to optimize joins using any
        // statistics their inputs might have.
        // - For a hash join with partition mode [`PartitionMode::Auto`], we will
        //   make a cost-based decision to select which `PartitionMode` mode
        //   (`Partitioned`/`CollectLeft`) is optimal. If the statistics information
        //   is not available, we will fall back to [`PartitionMode::Partitioned`].
        // - We optimize/swap join sides so that the left (build) side of the join
        //   is the small side. If the statistics information is not available, we
        //   do not modify join sides.
        // - We will also swap left and right sides for cross joins so that the left
        //   side is the small side.
        let config = &config.optimizer;
        let collect_threshold_byte_size = config.hash_join_single_partition_threshold;
        let collect_threshold_num_rows = config.hash_join_single_partition_threshold_rows;
        new_plan
            .transform_up(|plan| {
                statistical_join_selection_subrule(
                    plan,
                    collect_threshold_byte_size,
                    collect_threshold_num_rows,
                )
            })
            .data()
    }

    fn name(&self) -> &str {
        "join_selection"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Tries to create a [`HashJoinExec`] in [`PartitionMode::CollectLeft`] when possible.
///
/// This function will first consider the given join type and check whether the
/// `CollectLeft` mode is applicable. Otherwise, it will try to swap the join sides.
/// When the `ignore_threshold` is false, this function will also check left
/// and right sizes in bytes or rows.
pub(crate) fn try_collect_left(
    hash_join: &HashJoinExec,
    ignore_threshold: bool,
    threshold_byte_size: usize,
    threshold_num_rows: usize,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let left = hash_join.left();
    let right = hash_join.right();

    let left_can_collect = ignore_threshold
        || supports_collect_by_thresholds(
            &**left,
            threshold_byte_size,
            threshold_num_rows,
        );
    let right_can_collect = ignore_threshold
        || supports_collect_by_thresholds(
            &**right,
            threshold_byte_size,
            threshold_num_rows,
        );

    match (left_can_collect, right_can_collect) {
        (true, true) => {
            if hash_join.join_type().supports_swap()
                && should_swap_join_order(&**left, &**right)?
            {
                Ok(Some(hash_join.swap_inputs(PartitionMode::CollectLeft)?))
            } else {
                Ok(Some(Arc::new(HashJoinExec::try_new(
                    Arc::clone(left),
                    Arc::clone(right),
                    hash_join.on().to_vec(),
                    hash_join.filter().cloned(),
                    hash_join.join_type(),
                    hash_join.projection.clone(),
                    PartitionMode::CollectLeft,
                    hash_join.null_equality(),
                )?)))
            }
        }
        (true, false) => Ok(Some(Arc::new(HashJoinExec::try_new(
            Arc::clone(left),
            Arc::clone(right),
            hash_join.on().to_vec(),
            hash_join.filter().cloned(),
            hash_join.join_type(),
            hash_join.projection.clone(),
            PartitionMode::CollectLeft,
            hash_join.null_equality(),
        )?))),
        (false, true) => {
            if hash_join.join_type().supports_swap() {
                hash_join.swap_inputs(PartitionMode::CollectLeft).map(Some)
            } else {
                Ok(None)
            }
        }
        (false, false) => Ok(None),
    }
}

/// Creates a partitioned hash join execution plan, swapping inputs if beneficial.
///
/// Checks if the join order should be swapped based on the join type and input statistics.
/// If swapping is optimal and supported, creates a swapped partitioned hash join; otherwise,
/// creates a standard partitioned hash join.
pub(crate) fn partitioned_hash_join(
    hash_join: &HashJoinExec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let left = hash_join.left();
    let right = hash_join.right();
    if hash_join.join_type().supports_swap() && should_swap_join_order(&**left, &**right)?
    {
        hash_join.swap_inputs(PartitionMode::Partitioned)
    } else {
        Ok(Arc::new(HashJoinExec::try_new(
            Arc::clone(left),
            Arc::clone(right),
            hash_join.on().to_vec(),
            hash_join.filter().cloned(),
            hash_join.join_type(),
            hash_join.projection.clone(),
            PartitionMode::Partitioned,
            hash_join.null_equality(),
        )?))
    }
}

/// This subrule tries to modify a given plan so that it can
/// optimize hash and cross joins in the plan according to available statistical information.
fn statistical_join_selection_subrule(
    plan: Arc<dyn ExecutionPlan>,
    collect_threshold_byte_size: usize,
    collect_threshold_num_rows: usize,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let transformed = if let Some(asof_join) = try_asof_join(&plan)? {
        Some(asof_join)
    } else if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        match hash_join.partition_mode() {
            PartitionMode::Auto => try_collect_left(
                hash_join,
                false,
                collect_threshold_byte_size,
                collect_threshold_num_rows,
            )?
            .map_or_else(
                || partitioned_hash_join(hash_join).map(Some),
                |v| Ok(Some(v)),
            )?,
            PartitionMode::CollectLeft => try_collect_left(hash_join, true, 0, 0)?
                .map_or_else(
                    || partitioned_hash_join(hash_join).map(Some),
                    |v| Ok(Some(v)),
                )?,
            PartitionMode::Partitioned => {
                let left = hash_join.left();
                let right = hash_join.right();
                if hash_join.join_type().supports_swap()
                    && should_swap_join_order(&**left, &**right)?
                {
                    hash_join
                        .swap_inputs(PartitionMode::Partitioned)
                        .map(Some)?
                } else {
                    None
                }
            }
        }
    } else if let Some(cross_join) = plan.as_any().downcast_ref::<CrossJoinExec>() {
        let left = cross_join.left();
        let right = cross_join.right();
        if should_swap_join_order(&**left, &**right)? {
            cross_join.swap_inputs().map(Some)?
        } else {
            None
        }
    } else if let Some(nl_join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
        let left = nl_join.left();
        let right = nl_join.right();
        if nl_join.join_type().supports_swap()
            && should_swap_join_order(&**left, &**right)?
        {
            nl_join.swap_inputs().map(Some)?
        } else {
            None
        }
    } else {
        None
    };

    Ok(if let Some(transformed) = transformed {
        Transformed::yes(transformed)
    } else {
        Transformed::no(plan)
    })
}

fn try_asof_join(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if !matches!(hash_join.join_type(), JoinType::Inner | JoinType::Left) {
            return Ok(None);
        }

        let Some(filter) = hash_join.filter() else {
            return Ok(None);
        };
        let Some((filter_on, asof_condition)) =
            extract_asof_join_predicates(filter, hash_join.left(), hash_join.right())?
        else {
            return Ok(None);
        };

        let mut on = hash_join.on().to_vec();
        on.extend(filter_on);

        return Ok(Some(Arc::new(AsOfJoinExec::try_new(
            Arc::clone(hash_join.left()),
            Arc::clone(hash_join.right()),
            on,
            asof_condition,
            *hash_join.join_type(),
            hash_join.projection.clone(),
            hash_join.null_equality(),
        )?)));
    }

    if let Some(nl_join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
        if !matches!(nl_join.join_type(), JoinType::Inner | JoinType::Left) {
            return Ok(None);
        }

        let Some(filter) = nl_join.filter() else {
            return Ok(None);
        };
        let Some((on, asof_condition)) =
            extract_asof_join_predicates(filter, nl_join.left(), nl_join.right())?
        else {
            return Ok(None);
        };

        return Ok(Some(Arc::new(AsOfJoinExec::try_new(
            Arc::clone(nl_join.left()),
            Arc::clone(nl_join.right()),
            on,
            asof_condition,
            *nl_join.join_type(),
            nl_join.projection().cloned(),
            datafusion_common::NullEquality::NullEqualsNothing,
        )?)));
    }

    Ok(None)
}

fn extract_asof_join_predicates(
    filter: &JoinFilter,
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
) -> Result<Option<(JoinOn, AsOfJoinCondition)>> {
    let mut visitor = AsOfPredicateVisitor {
        filter,
        left,
        right,
        on: vec![],
        asof_condition: None,
        inequality_count: 0,
        unsupported: false,
    };
    visitor.visit(filter.expression())?;

    if visitor.unsupported || visitor.inequality_count != 1 {
        return Ok(None);
    }

    Ok(visitor
        .asof_condition
        .map(|asof_condition| (visitor.on, asof_condition)))
}

struct AsOfPredicateVisitor<'a> {
    filter: &'a JoinFilter,
    left: &'a Arc<dyn ExecutionPlan>,
    right: &'a Arc<dyn ExecutionPlan>,
    on: JoinOn,
    asof_condition: Option<AsOfJoinCondition>,
    inequality_count: usize,
    unsupported: bool,
}

impl AsOfPredicateVisitor<'_> {
    fn visit(&mut self, expr: &PhysicalExprRef) -> Result<()> {
        let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() else {
            self.unsupported = true;
            return Ok(());
        };

        if binary.op() == &Operator::And {
            self.visit(binary.left())?;
            self.visit(binary.right())?;
            return Ok(());
        }

        match binary.op() {
            Operator::Eq => {
                if let Some((left, right)) =
                    self.extract_side_pair(binary.left(), binary.right(), *binary.op())?
                {
                    self.on.push((left, right));
                } else {
                    self.unsupported = true;
                }
            }
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                self.inequality_count += 1;
                if self.inequality_count > 1 {
                    self.unsupported = true;
                    return Ok(());
                }

                if let Some((left, right, op)) =
                    self.extract_inequality(binary.left(), binary.right(), *binary.op())?
                {
                    self.asof_condition =
                        Some(AsOfJoinCondition::try_new(left, op, right)?);
                } else {
                    self.unsupported = true;
                }
            }
            _ => self.unsupported = true,
        }

        Ok(())
    }

    fn extract_side_pair(
        &self,
        left_expr: &PhysicalExprRef,
        right_expr: &PhysicalExprRef,
        op: Operator,
    ) -> Result<Option<(PhysicalExprRef, PhysicalExprRef)>> {
        let Some((left_side, left_expr)) = self.filter_column(left_expr)? else {
            return Ok(None);
        };
        let Some((right_side, right_expr)) = self.filter_column(right_expr)? else {
            return Ok(None);
        };

        match (left_side, right_side, op) {
            (JoinSide::Left, JoinSide::Right, Operator::Eq) => {
                Ok(Some((left_expr, right_expr)))
            }
            (JoinSide::Right, JoinSide::Left, Operator::Eq) => {
                Ok(Some((right_expr, left_expr)))
            }
            _ => Ok(None),
        }
    }

    fn extract_inequality(
        &self,
        left_expr: &PhysicalExprRef,
        right_expr: &PhysicalExprRef,
        op: Operator,
    ) -> Result<Option<(PhysicalExprRef, PhysicalExprRef, Operator)>> {
        let Some((left_side, left_expr)) = self.filter_column(left_expr)? else {
            return Ok(None);
        };
        let Some((right_side, right_expr)) = self.filter_column(right_expr)? else {
            return Ok(None);
        };

        match (left_side, right_side) {
            (JoinSide::Left, JoinSide::Right) => Ok(Some((left_expr, right_expr, op))),
            (JoinSide::Right, JoinSide::Left) => {
                Ok(Some((right_expr, left_expr, flip_inequality(op)?)))
            }
            _ => Ok(None),
        }
    }

    fn filter_column(
        &self,
        expr: &PhysicalExprRef,
    ) -> Result<Option<(JoinSide, PhysicalExprRef)>> {
        let Some(column) = expr.as_any().downcast_ref::<Column>() else {
            return Ok(None);
        };
        let Some(column_index) = self.filter.column_indices().get(column.index()) else {
            return Ok(None);
        };

        let (schema, side) = match column_index.side {
            JoinSide::Left => (self.left.schema(), JoinSide::Left),
            JoinSide::Right => (self.right.schema(), JoinSide::Right),
            JoinSide::None => return Ok(None),
        };

        let field = schema.field(column_index.index);
        Ok(Some((
            side,
            Arc::new(Column::new(field.name(), column_index.index)) as _,
        )))
    }
}

fn flip_inequality(op: Operator) -> Result<Operator> {
    match op {
        Operator::Lt => Ok(Operator::Gt),
        Operator::LtEq => Ok(Operator::GtEq),
        Operator::Gt => Ok(Operator::Lt),
        Operator::GtEq => Ok(Operator::LtEq),
        _ => internal_err!("Can not flip non-inequality operator {op}"),
    }
}

/// Pipeline-fixing join selection subrule.
pub type PipelineFixerSubrule =
    dyn Fn(Arc<dyn ExecutionPlan>, &ConfigOptions) -> Result<Arc<dyn ExecutionPlan>>;

/// Converts a hash join to a symmetric hash join if both its inputs are
/// unbounded and incremental.
///
/// This subrule checks if a hash join can be replaced with a symmetric hash join when dealing
/// with unbounded (infinite) inputs on both sides. This replacement avoids pipeline breaking and
/// preserves query runnability. If the replacement is applicable, this subrule makes this change;
/// otherwise, it leaves the input unchanged.
///
/// # Arguments
/// * `input` - The current state of the pipeline, including the execution plan.
/// * `config_options` - Configuration options that might affect the transformation logic.
///
/// # Returns
/// An `Option` that contains the `Result` of the transformation. If the transformation is not applicable,
/// it returns `None`. If applicable, it returns `Some(Ok(...))` with the modified pipeline state,
/// or `Some(Err(...))` if an error occurs during the transformation.
fn hash_join_convert_symmetric_subrule(
    input: Arc<dyn ExecutionPlan>,
    config_options: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Check if the current plan node is a HashJoinExec.
    if let Some(hash_join) = input.as_any().downcast_ref::<HashJoinExec>() {
        let left_unbounded = hash_join.left.boundedness().is_unbounded();
        let left_incremental = matches!(
            hash_join.left.pipeline_behavior(),
            EmissionType::Incremental | EmissionType::Both
        );
        let right_unbounded = hash_join.right.boundedness().is_unbounded();
        let right_incremental = matches!(
            hash_join.right.pipeline_behavior(),
            EmissionType::Incremental | EmissionType::Both
        );
        // Process only if both left and right sides are unbounded and incrementally emit.
        if left_unbounded && right_unbounded & left_incremental & right_incremental {
            // Determine the partition mode based on configuration.
            let mode = if config_options.optimizer.repartition_joins {
                StreamJoinPartitionMode::Partitioned
            } else {
                StreamJoinPartitionMode::SinglePartition
            };
            // A closure to determine the required sort order for each side of the join in the SymmetricHashJoinExec.
            // This function checks if the columns involved in the filter have any specific ordering requirements.
            // If the child nodes (left or right side of the join) already have a defined order and the columns used in the
            // filter predicate are ordered, this function captures that ordering requirement. The identified order is then
            // used in the SymmetricHashJoinExec to maintain bounded memory during join operations.
            // However, if the child nodes do not have an inherent order, or if the filter columns are unordered,
            // the function concludes that no specific order is required for the SymmetricHashJoinExec. This approach
            // ensures that the symmetric hash join operation only imposes ordering constraints when necessary,
            // based on the properties of the child nodes and the filter condition.
            let determine_order = |side: JoinSide| -> Option<LexOrdering> {
                hash_join
                    .filter()
                    .map(|filter| {
                        filter.column_indices().iter().any(
                            |ColumnIndex {
                                 index,
                                 side: column_side,
                             }| {
                                // Skip if column side does not match the join side.
                                if *column_side != side {
                                    return false;
                                }
                                // Retrieve equivalence properties and schema based on the side.
                                let (equivalence, schema) = match side {
                                    JoinSide::Left => (
                                        hash_join.left().equivalence_properties(),
                                        hash_join.left().schema(),
                                    ),
                                    JoinSide::Right => (
                                        hash_join.right().equivalence_properties(),
                                        hash_join.right().schema(),
                                    ),
                                    JoinSide::None => return false,
                                };

                                let name = schema.field(*index).name();
                                let col = Arc::new(Column::new(name, *index)) as _;
                                // Check if the column is ordered.
                                equivalence.get_expr_properties(col).sort_properties
                                    != SortProperties::Unordered
                            },
                        )
                    })
                    .unwrap_or(false)
                    .then(|| {
                        match side {
                            JoinSide::Left => hash_join.left().output_ordering(),
                            JoinSide::Right => hash_join.right().output_ordering(),
                            JoinSide::None => unreachable!(),
                        }
                        .cloned()
                    })
                    .flatten()
            };

            // Determine the sort order for both left and right sides.
            let left_order = determine_order(JoinSide::Left);
            let right_order = determine_order(JoinSide::Right);

            return SymmetricHashJoinExec::try_new(
                Arc::clone(hash_join.left()),
                Arc::clone(hash_join.right()),
                hash_join.on().to_vec(),
                hash_join.filter().cloned(),
                hash_join.join_type(),
                hash_join.null_equality(),
                left_order,
                right_order,
                mode,
            )
            .map(|exec| Arc::new(exec) as _);
        }
    }
    Ok(input)
}

/// This subrule will swap build/probe sides of a hash join depending on whether
/// one of its inputs may produce an infinite stream of records. The rule ensures
/// that the left (build) side of the hash join always operates on an input stream
/// that will produce a finite set of records. If the left side can not be chosen
/// to be "finite", the join sides stay the same as the original query.
/// ```text
/// For example, this rule makes the following transformation:
///
///
///
///           +--------------+              +--------------+
///           |              |  unbounded   |              |
///    Left   | Infinite     |    true      | Hash         |\true
///           | Data source  |--------------| Repartition  | \   +--------------+       +--------------+
///           |              |              |              |  \  |              |       |              |
///           +--------------+              +--------------+   - |  Hash Join   |-------| Projection   |
///                                                            - |              |       |              |
///           +--------------+              +--------------+  /  +--------------+       +--------------+
///           |              |  unbounded   |              | /
///    Right  | Finite       |    false     | Hash         |/false
///           | Data Source  |--------------| Repartition  |
///           |              |              |              |
///           +--------------+              +--------------+
///
///
///
///           +--------------+              +--------------+
///           |              |  unbounded   |              |
///    Left   | Finite       |    false     | Hash         |\false
///           | Data source  |--------------| Repartition  | \   +--------------+       +--------------+
///           |              |              |              |  \  |              | true  |              | true
///           +--------------+              +--------------+   - |  Hash Join   |-------| Projection   |-----
///                                                            - |              |       |              |
///           +--------------+              +--------------+  /  +--------------+       +--------------+
///           |              |  unbounded   |              | /
///    Right  | Infinite     |    true      | Hash         |/true
///           | Data Source  |--------------| Repartition  |
///           |              |              |              |
///           +--------------+              +--------------+
///
/// ```
pub fn hash_join_swap_subrule(
    mut input: Arc<dyn ExecutionPlan>,
    _config_options: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(hash_join) = input.as_any().downcast_ref::<HashJoinExec>() {
        if hash_join.left.boundedness().is_unbounded()
            && !hash_join.right.boundedness().is_unbounded()
            && matches!(
                *hash_join.join_type(),
                JoinType::Inner
                    | JoinType::Left
                    | JoinType::LeftSemi
                    | JoinType::LeftAnti
            )
        {
            input = swap_join_according_to_unboundedness(hash_join)?;
        }
    }
    Ok(input)
}

/// This function swaps sides of a hash join to make it runnable even if one of
/// its inputs are infinite. Note that this is not always possible; i.e.
/// [`JoinType::Full`], [`JoinType::Right`], [`JoinType::RightAnti`] and
/// [`JoinType::RightSemi`] can not run with an unbounded left side, even if
/// we swap join sides. Therefore, we do not consider them here.
/// This function is crate public as it is useful for downstream projects
/// to implement, or experiment with, their own join selection rules.
pub(crate) fn swap_join_according_to_unboundedness(
    hash_join: &HashJoinExec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let partition_mode = hash_join.partition_mode();
    let join_type = hash_join.join_type();
    match (*partition_mode, *join_type) {
        (
            _,
            JoinType::Right | JoinType::RightSemi | JoinType::RightAnti | JoinType::Full,
        ) => internal_err!("{join_type} join cannot be swapped for unbounded input."),
        (PartitionMode::Partitioned, _) => {
            hash_join.swap_inputs(PartitionMode::Partitioned)
        }
        (PartitionMode::CollectLeft, _) => {
            hash_join.swap_inputs(PartitionMode::CollectLeft)
        }
        (PartitionMode::Auto, _) => {
            // Use `PartitionMode::Partitioned` as default if `Auto` is selected.
            hash_join.swap_inputs(PartitionMode::Partitioned)
        }
    }
}

/// Apply given `PipelineFixerSubrule`s to a given plan. This plan, along with
/// auxiliary boundedness information, is in the `PipelineStatePropagator` object.
fn apply_subrules(
    mut input: Arc<dyn ExecutionPlan>,
    subrules: &Vec<Box<PipelineFixerSubrule>>,
    config_options: &ConfigOptions,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    for subrule in subrules {
        input = subrule(input, config_options)?;
    }
    Ok(Transformed::yes(input))
}

// See tests in datafusion/core/tests/physical_optimizer
