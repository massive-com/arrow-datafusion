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

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use datafusion_common::{
    Column, NullEquality, Result, not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::utils::{find_valid_equijoin_key_pair, split_conjunction};
use datafusion_expr::{
    AsOfJoin, BinaryExpr, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator,
};
use sqlparser::ast::{
    Expr as SqlExpr, Join, JoinConstraint, JoinOperator, ObjectName, TableFactor,
    TableWithJoins,
};
use std::collections::HashSet;
use std::sync::Arc;

impl<S: ContextProvider> SqlToRel<'_, S> {
    pub(crate) fn plan_table_with_joins(
        &self,
        t: TableWithJoins,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let mut left = if is_lateral(&t.relation) {
            self.create_relation_subquery(t.relation, planner_context)?
        } else {
            self.create_relation(t.relation, planner_context)?
        };
        let old_outer_from_schema = planner_context.outer_from_schema();
        for join in t.joins {
            planner_context.extend_outer_from_schema(left.schema())?;
            left = self.parse_relation_join(left, join, planner_context)?;
        }
        planner_context.set_outer_from_schema(old_outer_from_schema);
        Ok(left)
    }

    pub(crate) fn parse_relation_join(
        &self,
        left: LogicalPlan,
        join: Join,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let right = if is_lateral_join(&join)? {
            self.create_relation_subquery(join.relation, planner_context)?
        } else {
            self.create_relation(join.relation, planner_context)?
        };
        match join.join_operator {
            JoinOperator::LeftOuter(constraint) | JoinOperator::Left(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Left, planner_context)
            }
            JoinOperator::RightOuter(constraint) | JoinOperator::Right(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Right, planner_context)
            }
            JoinOperator::Inner(constraint) | JoinOperator::Join(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Inner, planner_context)
            }
            JoinOperator::LeftSemi(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::LeftSemi,
                planner_context,
            ),
            JoinOperator::RightSemi(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::RightSemi,
                planner_context,
            ),
            JoinOperator::LeftAnti(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::LeftAnti,
                planner_context,
            ),
            JoinOperator::RightAnti(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::RightAnti,
                planner_context,
            ),
            JoinOperator::FullOuter(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Full, planner_context)
            }
            JoinOperator::CrossJoin(JoinConstraint::None) => {
                self.parse_cross_join(left, right)
            }
            JoinOperator::AsOf {
                match_condition,
                constraint,
            } => self.parse_asof_join(
                left,
                right,
                match_condition,
                constraint,
                planner_context,
            ),
            other => not_impl_err!("Unsupported JOIN operator {other:?}"),
        }
    }

    fn parse_cross_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(left).cross_join(right)?.build()
    }

    fn parse_asof_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        match_condition: SqlExpr,
        constraint: JoinConstraint,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let join_schema = left.schema().join(right.schema())?;

        // The MATCH_CONDITION must be a single inequality comparison. The
        // physical planner auto-orients its operands, so we only validate here.
        let match_condition =
            self.sql_to_expr(match_condition, &join_schema, planner_context)?;
        match &match_condition {
            Expr::BinaryExpr(BinaryExpr { op, .. })
                if matches!(
                    op,
                    Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
                ) => {}
            other => {
                return plan_err!(
                    "ASOF JOIN MATCH_CONDITION must be a single inequality \
                     (<, <=, >, >=), got {other}"
                );
            }
        }

        // Build the equijoin key pairs from the ON clause, oriented so that
        // `.0` references the left input and `.1` the right input.
        let on = match constraint {
            JoinConstraint::On(sql_expr) => {
                let on_expr =
                    self.sql_to_expr(sql_expr, &join_schema, planner_context)?;
                let mut keys: Vec<(Expr, Expr)> = vec![];
                for conjunct in split_conjunction(&on_expr) {
                    let Expr::BinaryExpr(BinaryExpr {
                        left: lhs,
                        op: Operator::Eq,
                        right: rhs,
                    }) = conjunct
                    else {
                        return plan_err!(
                            "ASOF JOIN ON clause must be equality conditions \
                             between the two inputs"
                        );
                    };
                    match find_valid_equijoin_key_pair(
                        lhs,
                        rhs,
                        left.schema(),
                        right.schema(),
                    )? {
                        Some(pair) => keys.push(pair),
                        None => {
                            return plan_err!(
                                "ASOF JOIN ON clause must be equality conditions \
                                 between the two inputs"
                            );
                        }
                    }
                }
                keys
            }
            // ASOF with no partition keys is allowed.
            JoinConstraint::None => vec![],
            JoinConstraint::Using(_) | JoinConstraint::Natural => {
                return not_impl_err!("USING/NATURAL not supported for ASOF JOIN");
            }
        };

        // Snowflake ASOF keeps every left row, emitting NULLs when there is no
        // matching right row, so default to a LEFT join.
        Ok(LogicalPlan::AsOfJoin(AsOfJoin::try_new(
            Arc::new(left),
            Arc::new(right),
            on,
            match_condition,
            JoinType::Left,
            NullEquality::NullEqualsNothing,
        )?))
    }

    fn parse_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        constraint: JoinConstraint,
        join_type: JoinType,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        match constraint {
            JoinConstraint::On(sql_expr) => {
                let join_schema = left.schema().join(right.schema())?;
                // parse ON expression
                let expr = self.sql_to_expr(sql_expr, &join_schema, planner_context)?;
                LogicalPlanBuilder::from(left)
                    .join_on(right, join_type, Some(expr))?
                    .build()
            }
            JoinConstraint::Using(object_names) => {
                let keys = object_names
                    .into_iter()
                    .map(|object_name| {
                        let ObjectName(mut object_names) = object_name;
                        if object_names.len() != 1 {
                            not_impl_err!(
                                "Invalid identifier in USING clause. Expected single identifier, got {}", ObjectName(object_names)
                            )
                        } else {
                            let id = object_names.swap_remove(0);
                            id.as_ident()
                                .ok_or_else(|| {
                                    plan_datafusion_err!(
                                        "Expected identifier in USING clause"
                                    )
                                })
                                .map(|ident| Column::from_name(self.ident_normalizer.normalize(ident.clone())))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                LogicalPlanBuilder::from(left)
                    .join_using(right, join_type, keys)?
                    .build()
            }
            JoinConstraint::Natural => {
                let left_cols: HashSet<&String> =
                    left.schema().fields().iter().map(|f| f.name()).collect();
                let keys: Vec<Column> = right
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .filter(|f| left_cols.contains(f))
                    .map(Column::from_name)
                    .collect();
                if keys.is_empty() {
                    self.parse_cross_join(left, right)
                } else {
                    LogicalPlanBuilder::from(left)
                        .join_using(right, join_type, keys)?
                        .build()
                }
            }
            JoinConstraint::None => LogicalPlanBuilder::from(left)
                .join_on(right, join_type, [])?
                .build(),
        }
    }
}

/// Return `true` iff the given [`TableFactor`] is lateral.
pub(crate) fn is_lateral(factor: &TableFactor) -> bool {
    match factor {
        TableFactor::Derived { lateral, .. } => *lateral,
        TableFactor::Function { lateral, .. } => *lateral,
        TableFactor::UNNEST { .. } => true,
        _ => false,
    }
}

/// Return `true` iff the given [`Join`] is lateral.
pub(crate) fn is_lateral_join(join: &Join) -> Result<bool> {
    let is_lateral_syntax = is_lateral(&join.relation);
    let is_apply_syntax = match join.join_operator {
        JoinOperator::FullOuter(..)
        | JoinOperator::Right(..)
        | JoinOperator::RightOuter(..)
        | JoinOperator::RightAnti(..)
        | JoinOperator::RightSemi(..)
            if is_lateral_syntax =>
        {
            return not_impl_err!(
                "LATERAL syntax is not supported for \
                 FULL OUTER and RIGHT [OUTER | ANTI | SEMI] joins"
            );
        }
        JoinOperator::CrossApply | JoinOperator::OuterApply => true,
        _ => false,
    };
    Ok(is_lateral_syntax || is_apply_syntax)
}
