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

//! Grouped `ARRAY_AGG(DISTINCT ...)`: [`DistinctArrayAggGroupsAccumulator`]

use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, ListArray,
    NullBufferBuilder, PrimitiveArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::{SortOptions, cast, sort_to_indices};
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type,
    Decimal256Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Field, Int8Type, Int16Type, Int32Type,
    Int64Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalUnit,
    IntervalYearMonthType, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::cast::as_list_array;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::utils::proxy::HashTableAllocExt;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_common::{HashMap, Result, exec_datafusion_err, exec_err, internal_err};
use datafusion_expr::{EmitTo, GroupSelection, GroupsAccumulator};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::filter_to_nulls;
use datafusion_physical_expr_common::binary_map::{ArrowBytesMap, OutputType};
use datafusion_physical_expr_common::binary_view_map::ArrowBytesViewMap;
use hashbrown::hash_table::HashTable;

/// Largest group count the packed `(group, value id)` representation can address.
const MAX_GROUPS: usize = u32::MAX as usize;

/// Largest number of distinct values the dictionary can hold.
const MAX_VALUES: usize = u32::MAX as usize;

/// [`GroupsAccumulator`] for `ARRAY_AGG(DISTINCT ...)`.
///
/// All groups share one dictionary of distinct values, and a group's content is
/// the set of `(group, value id)` pairs it owns. Compared to one hash set per
/// group this costs a few bytes per pair instead of a copy of the value, and
/// `state` and `evaluate` materialize every group in a single pass.
///
/// Without `ORDER BY` each list holds its values in the order the group first
/// saw them. With `ORDER BY` (which must sort by the aggregated expression
/// itself) each list is sorted by value.
#[derive(Debug)]
pub struct DistinctArrayAggGroupsAccumulator {
    /// Declared element type of the output lists.
    data_type: DataType,
    /// Type the dictionary interns: `data_type` without a top-level `Dictionary`.
    storage_type: DataType,
    /// Sort direction of `ORDER BY`, if present.
    sort_options: Option<SortOptions>,
    ignore_nulls: bool,
    /// Distinct values of all groups. A value's id is its insertion index.
    dictionary: ValueDictionary,
    /// Distinct `(group, value id)` pairs in insertion order, see [`pack`].
    pairs: Vec<u64>,
    /// `pairs` as a set, for deduplication.
    pair_set: HashTable<u64>,
    /// Bytes allocated by `pair_set`.
    pair_set_size: usize,
    /// Distinct value count per group; its length is the number of groups.
    counts: Vec<u32>,
    random_state: RandomState,
    /// Scratch buffer: value id of every row of the batch being interned.
    ids: Vec<u32>,
}

impl DistinctArrayAggGroupsAccumulator {
    /// Returns whether values of `data_type` can be aggregated by this accumulator.
    pub fn supports_type(data_type: &DataType) -> bool {
        ValueDictionary::new(&storage_type_of(data_type)).is_some()
    }

    /// Creates an accumulator for elements of `data_type`.
    ///
    /// `sort_options` is the direction of an `ORDER BY` on the aggregated
    /// expression. Returns an error for types [`Self::supports_type`] rejects.
    pub fn try_new(
        data_type: &DataType,
        sort_options: Option<SortOptions>,
        ignore_nulls: bool,
    ) -> Result<Self> {
        let storage_type = storage_type_of(data_type);
        let Some(dictionary) = ValueDictionary::new(&storage_type) else {
            return internal_err!(
                "ARRAY_AGG(DISTINCT) groups accumulator does not support {data_type}"
            );
        };
        Ok(Self {
            data_type: data_type.clone(),
            storage_type,
            sort_options,
            ignore_nulls,
            dictionary,
            pairs: Vec::new(),
            pair_set: HashTable::new(),
            pair_set_size: 0,
            counts: Vec::new(),
            random_state: RandomState::default(),
            ids: Vec::new(),
        })
    }

    /// Number of groups seen so far.
    fn num_groups(&self) -> usize {
        self.counts.len()
    }

    fn grow_groups(&mut self, total_num_groups: usize) -> Result<()> {
        if total_num_groups > MAX_GROUPS {
            return exec_err!(
                "ARRAY_AGG(DISTINCT) supports at most {MAX_GROUPS} groups per partition"
            );
        }
        if total_num_groups > self.counts.len() {
            self.counts.resize(total_num_groups, 0);
        }
        Ok(())
    }

    /// Interns `values` into the dictionary, leaving each row's id in `self.ids`.
    fn intern(&mut self, values: &ArrayRef) -> Result<()> {
        let converted;
        let values = if values.data_type() == &self.storage_type {
            values
        } else {
            converted = cast(values.as_ref(), &self.storage_type)?;
            &converted
        };
        self.dictionary.intern(values, &mut self.ids)
    }

    /// Id of the NULL value when it must be skipped, so callers can compare
    /// ids instead of consulting a null buffer per row.
    fn skipped_null_id(&self) -> Option<u32> {
        self.ignore_nulls
            .then(|| self.dictionary.null_id())
            .flatten()
    }

    /// Records `value` for `group` unless the pair is already known.
    #[inline]
    fn insert_pair(&mut self, group: usize, value: u32) {
        let key = pack(group, value);
        let random_state = &self.random_state;
        let hash = random_state.hash_one(key);
        if self.pair_set.find(hash, |&known| known == key).is_some() {
            return;
        }
        self.pair_set.insert_accounted(
            key,
            |&key| random_state.hash_one(key),
            &mut self.pair_set_size,
        );
        self.pairs.push(key);
        self.counts[group] += 1;
    }

    /// Builds one list per output slot.
    ///
    /// `output_counts[slot]` is the number of values `slot` receives, and
    /// `visit` feeds every `(slot, value id)` in insertion order.
    fn build_lists(
        &self,
        output_counts: &[u32],
        visit: impl FnOnce(&mut dyn FnMut(usize, u32)),
    ) -> Result<ArrayRef> {
        let mut offsets = Vec::with_capacity(output_counts.len() + 1);
        offsets.push(0i32);
        let mut nulls = NullBufferBuilder::new(output_counts.len());
        let mut cursors = Vec::with_capacity(output_counts.len());
        let mut total = 0usize;
        for &count in output_counts {
            nulls.append(count > 0);
            cursors.push(total);
            total += count as usize;
            offsets.push(i32::try_from(total).map_err(|_| {
                exec_datafusion_err!(
                    "ARRAY_AGG(DISTINCT) output exceeds i32 offset capacity"
                )
            })?);
        }

        let mut ids = vec![0u32; total];
        visit(&mut |slot, id| {
            ids[cursors[slot]] = id;
            cursors[slot] += 1;
        });

        if let Some(options) = self.sort_options {
            let ranks = self.value_ranks(options)?;
            for window in offsets.windows(2) {
                let (start, end) = (window[0] as usize, window[1] as usize);
                ids[start..end].sort_unstable_by_key(|&id| ranks[id as usize]);
            }
        }

        let values = self.dictionary.keys(&ids)?;
        let field = Arc::new(Field::new_list_field(self.storage_type.clone(), true));
        let lists = ListArray::try_new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            values,
            nulls.finish(),
        )?;
        if self.storage_type == self.data_type {
            return Ok(Arc::new(lists));
        }
        let list_type = DataType::List(Arc::new(Field::new_list_field(
            self.data_type.clone(),
            true,
        )));
        Ok(cast(&lists, &list_type)?)
    }

    /// Rank of every dictionary value under `options`, indexed by value id.
    fn value_ranks(&self, options: SortOptions) -> Result<Vec<u32>> {
        let all_ids: Vec<u32> = (0..self.dictionary.len() as u32).collect();
        let values = self.dictionary.keys(&all_ids)?;
        let sorted = sort_to_indices(values.as_ref(), Some(options), None)?;
        let mut ranks = vec![0u32; all_ids.len()];
        for (rank, &id) in sorted.values().iter().enumerate() {
            ranks[id as usize] = rank as u32;
        }
        Ok(ranks)
    }

    /// Drops all state and releases its allocations.
    fn reset(&mut self) {
        self.dictionary.clear();
        self.pairs = Vec::new();
        self.pair_set = HashTable::new();
        self.pair_set_size = 0;
        self.counts = Vec::new();
        self.ids = Vec::new();
    }

    /// Drops groups `0..n` and renumbers the rest from zero.
    ///
    /// The dictionary is rebuilt from the retained pairs so that it only holds
    /// values the retained groups still reference.
    fn retain_groups_from(&mut self, n: usize) -> Result<()> {
        let retained: Vec<u64> = self
            .pairs
            .iter()
            .copied()
            .filter(|&key| unpack(key).0 >= n)
            .collect();
        let retained_ids: Vec<u32> = retained.iter().map(|&key| unpack(key).1).collect();
        let retained_values = self.dictionary.keys(&retained_ids)?;
        self.dictionary.clear();
        self.dictionary.intern(&retained_values, &mut self.ids)?;

        self.pairs = Vec::with_capacity(retained.len());
        self.pair_set = HashTable::with_capacity(retained.len());
        self.pair_set_size = self.pair_set.capacity() * size_of::<u64>();
        let random_state = &self.random_state;
        for (&key, &id) in retained.iter().zip(&self.ids) {
            let key = pack(unpack(key).0 - n, id);
            self.pair_set
                .insert_unique(random_state.hash_one(key), key, |&key| {
                    random_state.hash_one(key)
                });
            self.pairs.push(key);
        }
        split_vec_min_alloc(&mut self.counts, n);
        Ok(())
    }
}

impl GroupsAccumulator for DistinctArrayAggGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // With ORDER BY the sort expression follows the argument; it is the
        // same expression, so only the first column is needed.
        let Some(values) = values.first() else {
            return internal_err!("ARRAY_AGG(DISTINCT) expects an argument");
        };
        self.grow_groups(total_num_groups)?;
        self.intern(values)?;
        let skipped_null_id = self.skipped_null_id();

        for (row, &group) in group_indices.iter().enumerate() {
            if let Some(filter) = opt_filter
                && !(filter.is_valid(row) && filter.value(row))
            {
                continue;
            }
            let id = self.ids[row];
            if Some(id) == skipped_null_id {
                continue;
            }
            self.insert_pair(group, id);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let num_emitted = match emit_to {
            EmitTo::All => self.num_groups(),
            EmitTo::First(n) => n,
        };
        if num_emitted > self.num_groups() {
            return internal_err!(
                "ARRAY_AGG(DISTINCT) asked to emit {num_emitted} of {} groups",
                self.num_groups()
            );
        }

        let lists = self.build_lists(&self.counts[..num_emitted], |sink| {
            for &key in &self.pairs {
                let (group, id) = unpack(key);
                if group < num_emitted {
                    sink(group, id);
                }
            }
        })?;

        match emit_to {
            EmitTo::All => self.reset(),
            EmitTo::First(n) => self.retain_groups_from(n)?,
        }
        Ok(lists)
    }

    fn evaluate_preserving(&mut self, selection: GroupSelection<'_>) -> Result<ArrayRef> {
        selection.validate_num_groups(self.num_groups())?;
        let mut slots_by_group: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut output_counts = Vec::with_capacity(selection.len());
        for (slot, group) in selection.iter().enumerate() {
            slots_by_group.entry(group).or_default().push(slot);
            output_counts.push(self.counts[group]);
        }

        self.build_lists(&output_counts, |sink| {
            for &key in &self.pairs {
                let (group, id) = unpack(key);
                if let Some(slots) = slots_by_group.get(&group) {
                    for &slot in slots {
                        sink(slot, id);
                    }
                }
            }
        })
    }

    fn supports_evaluate_preserving(&self) -> bool {
        true
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        Ok(vec![self.evaluate(emit_to)?])
    }

    fn state_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        Ok(vec![self.evaluate_preserving(selection)?])
    }

    fn supports_state_preserving(&self) -> bool {
        true
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        let [states] = values else {
            return internal_err!("ARRAY_AGG(DISTINCT) expects a single state column");
        };
        let lists = as_list_array(states)?;
        self.grow_groups(total_num_groups)?;

        // A sliced list array still carries the whole backing values array;
        // only intern the range this slice references.
        let offsets = lists.offsets();
        let first = offsets[0] as usize;
        let last = offsets[offsets.len() - 1] as usize;
        self.intern(&lists.values().slice(first, last - first))?;
        let skipped_null_id = self.skipped_null_id();

        for (row, &group) in group_indices.iter().enumerate() {
            if lists.is_null(row) {
                continue;
            }
            for position in offsets[row] as usize..offsets[row + 1] as usize {
                let id = self.ids[position - first];
                if Some(id) == skipped_null_id {
                    continue;
                }
                self.insert_pair(group, id);
            }
        }
        Ok(())
    }

    /// Wraps every row in a one-element list, the same layout `state` emits.
    ///
    /// Filtered rows become null lists, as do null values when nulls are
    /// ignored; `merge_batch` skips both.
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let Some(input) = values.first() else {
            return internal_err!("ARRAY_AGG(DISTINCT) expects an argument");
        };
        let offsets = OffsetBuffer::from_repeated_length(1, input.len());
        let filter_nulls = opt_filter.map(filter_to_nulls);
        let nulls = if self.ignore_nulls {
            NullBuffer::union(filter_nulls.as_ref(), input.logical_nulls().as_ref())
        } else {
            filter_nulls
        };
        let field = Arc::new(Field::new_list_field(self.data_type.clone(), true));
        let lists = ListArray::try_new(field, offsets, Arc::clone(input), nulls)?;
        Ok(vec![Arc::new(lists)])
    }

    fn size(&self) -> usize {
        size_of::<Self>()
            + self.dictionary.size()
            + self.pairs.capacity() * size_of::<u64>()
            + self.pair_set_size
            + self.counts.capacity() * size_of::<u32>()
            + self.ids.capacity() * size_of::<u32>()
    }
}

/// Packs a `(group, value id)` pair into the key stored in `pairs`.
#[inline]
fn pack(group: usize, value: u32) -> u64 {
    ((group as u64) << 32) | value as u64
}

/// Inverse of [`pack`].
#[inline]
fn unpack(key: u64) -> (usize, u32) {
    ((key >> 32) as usize, key as u32)
}

/// The type whose values the dictionary interns for elements of `data_type`.
fn storage_type_of(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value_type) => value_type.as_ref().clone(),
        other => other.clone(),
    }
}

/// Distinct values shared by all groups.
///
/// Ids are assigned in first-seen order and double as positions in the arrays
/// [`Self::keys`] builds. NULL, if seen, gets an id like any other value.
#[derive(Debug)]
struct ValueDictionary {
    kind: DictionaryKind,
    /// Id of the NULL value, once seen.
    null_id: Option<u32>,
}

#[derive(Debug)]
enum DictionaryKind {
    Bytes(ArrowBytesMap<i32, u32>),
    LargeBytes(ArrowBytesMap<i64, u32>),
    BytesView(ArrowBytesViewMap<u32>),
    Native(Box<dyn NativeDictionary>),
}

impl ValueDictionary {
    /// Returns `None` for types without a dictionary implementation.
    fn new(storage_type: &DataType) -> Option<Self> {
        let kind = match storage_type {
            DataType::Utf8 => DictionaryKind::Bytes(ArrowBytesMap::new(OutputType::Utf8)),
            DataType::LargeUtf8 => {
                DictionaryKind::LargeBytes(ArrowBytesMap::new(OutputType::Utf8))
            }
            DataType::Binary => {
                DictionaryKind::Bytes(ArrowBytesMap::new(OutputType::Binary))
            }
            DataType::LargeBinary => {
                DictionaryKind::LargeBytes(ArrowBytesMap::new(OutputType::Binary))
            }
            DataType::Utf8View => {
                DictionaryKind::BytesView(ArrowBytesViewMap::new(OutputType::Utf8View))
            }
            DataType::BinaryView => {
                DictionaryKind::BytesView(ArrowBytesViewMap::new(OutputType::BinaryView))
            }
            other => DictionaryKind::Native(native_dictionary(other)?),
        };
        Some(Self {
            kind,
            null_id: None,
        })
    }

    /// Interns every row of `values`, which must be of the storage type, and
    /// writes each row's id to `ids`.
    fn intern(&mut self, values: &ArrayRef, ids: &mut Vec<u32>) -> Result<()> {
        if self.len().saturating_add(values.len()) > MAX_VALUES {
            return exec_err!(
                "ARRAY_AGG(DISTINCT) supports at most {MAX_VALUES} distinct values per partition"
            );
        }
        ids.clear();

        let mut next_id = self.len() as u32;
        let null_id = &mut self.null_id;
        let make_id = |value: Option<&[u8]>| {
            let id = next_id;
            next_id += 1;
            if value.is_none() {
                *null_id = Some(id);
            }
            id
        };
        match &mut self.kind {
            DictionaryKind::Bytes(map) => {
                map.insert_if_new(values, make_id, |id| ids.push(id))
            }
            DictionaryKind::LargeBytes(map) => {
                map.insert_if_new(values, make_id, |id| ids.push(id))
            }
            DictionaryKind::BytesView(map) => {
                map.insert_if_new(values, make_id, |id| ids.push(id))
            }
            DictionaryKind::Native(dictionary) => dictionary.intern(values, ids, null_id),
        }
        Ok(())
    }

    /// Number of ids handed out, including the NULL id.
    fn len(&self) -> usize {
        match &self.kind {
            DictionaryKind::Bytes(map) => map.len(),
            DictionaryKind::LargeBytes(map) => map.len(),
            DictionaryKind::BytesView(map) => map.len(),
            DictionaryKind::Native(dictionary) => dictionary.len(),
        }
    }

    fn null_id(&self) -> Option<u32> {
        self.null_id
    }

    /// Copies the values with the given ids into an array, in order.
    fn keys(&self, ids: &[u32]) -> Result<ArrayRef> {
        match &self.kind {
            DictionaryKind::Bytes(map) => map.keys(ids.iter().map(|&id| id as usize)),
            DictionaryKind::LargeBytes(map) => {
                map.keys(ids.iter().map(|&id| id as usize))
            }
            DictionaryKind::BytesView(map) => map.keys(ids.iter().map(|&id| id as usize)),
            DictionaryKind::Native(dictionary) => dictionary.keys(ids, self.null_id),
        }
    }

    /// Forgets all values, releasing their memory.
    fn clear(&mut self) {
        match &mut self.kind {
            DictionaryKind::Bytes(map) => drop(map.take()),
            DictionaryKind::LargeBytes(map) => drop(map.take()),
            DictionaryKind::BytesView(map) => drop(map.take()),
            DictionaryKind::Native(dictionary) => dictionary.clear(),
        }
        self.null_id = None;
    }

    fn size(&self) -> usize {
        match &self.kind {
            DictionaryKind::Bytes(map) => map.size(),
            DictionaryKind::LargeBytes(map) => map.size(),
            DictionaryKind::BytesView(map) => map.size(),
            DictionaryKind::Native(dictionary) => dictionary.size(),
        }
    }
}

/// Dictionary of fixed-width values, type-erased so the accumulator needs no
/// type parameter.
trait NativeDictionary: Send + Debug {
    /// Interns every row of `values`, appending each row's id to `ids` and
    /// recording the id of NULL in `null_id` when first seen.
    fn intern(
        &mut self,
        values: &ArrayRef,
        ids: &mut Vec<u32>,
        null_id: &mut Option<u32>,
    );

    fn len(&self) -> usize;

    /// Copies the values with the given ids into an array, emitting NULL for
    /// `null_id`.
    fn keys(&self, ids: &[u32], null_id: Option<u32>) -> Result<ArrayRef>;

    fn clear(&mut self);

    fn size(&self) -> usize;
}

/// [`NativeDictionary`] for one Arrow primitive type.
struct PrimitiveDictionary<T: ArrowPrimitiveType> {
    /// Full output type, which may carry a time zone or decimal precision.
    data_type: DataType,
    ids: HashMap<T::Native, u32>,
    /// Values by id; the NULL id holds a placeholder.
    values: Vec<T::Native>,
}

impl<T: ArrowPrimitiveType> PrimitiveDictionary<T> {
    fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            ids: HashMap::default(),
            values: Vec::new(),
        }
    }
}

impl<T: ArrowPrimitiveType> Debug for PrimitiveDictionary<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveDictionary")
            .field("data_type", &self.data_type)
            .field("len", &self.values.len())
            .finish()
    }
}

impl<T: ArrowPrimitiveType> NativeDictionary for PrimitiveDictionary<T>
where
    T::Native: Hash + Eq,
{
    fn intern(
        &mut self,
        values: &ArrayRef,
        ids: &mut Vec<u32>,
        null_id: &mut Option<u32>,
    ) {
        let Self {
            ids: known,
            values: stored,
            ..
        } = self;
        for value in values.as_primitive::<T>().iter() {
            let id = match value {
                Some(value) => *known.entry(value).or_insert_with(|| {
                    stored.push(value);
                    (stored.len() - 1) as u32
                }),
                None => *null_id.get_or_insert_with(|| {
                    stored.push(T::Native::default());
                    (stored.len() - 1) as u32
                }),
            };
            ids.push(id);
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn keys(&self, ids: &[u32], null_id: Option<u32>) -> Result<ArrayRef> {
        let mut values = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(&value) = self.values.get(id as usize) else {
                return exec_err!(
                    "Value id {id} is out of bounds for {} values",
                    self.values.len()
                );
            };
            values.push(value);
        }
        let nulls = null_id
            .map(|null_id| NullBuffer::from_iter(ids.iter().map(|&id| id != null_id)));
        let array = PrimitiveArray::<T>::try_new(ScalarBuffer::from(values), nulls)?
            .with_data_type(self.data_type.clone());
        Ok(Arc::new(array))
    }

    fn clear(&mut self) {
        self.ids = HashMap::default();
        self.values = Vec::new();
    }

    fn size(&self) -> usize {
        self.ids.capacity() * (size_of::<T::Native>() + size_of::<u32>() + 1)
            + self.values.capacity() * size_of::<T::Native>()
    }
}

/// A [`NativeDictionary`] for `data_type`, or `None` if its values cannot be
/// hashed exactly (floats) or are not primitive.
fn native_dictionary(data_type: &DataType) -> Option<Box<dyn NativeDictionary>> {
    macro_rules! dictionary {
        ($t:ty) => {
            Box::new(PrimitiveDictionary::<$t>::new(data_type.clone()))
                as Box<dyn NativeDictionary>
        };
    }
    Some(match data_type {
        DataType::Int8 => dictionary!(Int8Type),
        DataType::Int16 => dictionary!(Int16Type),
        DataType::Int32 => dictionary!(Int32Type),
        DataType::Int64 => dictionary!(Int64Type),
        DataType::UInt8 => dictionary!(UInt8Type),
        DataType::UInt16 => dictionary!(UInt16Type),
        DataType::UInt32 => dictionary!(UInt32Type),
        DataType::UInt64 => dictionary!(UInt64Type),
        DataType::Date32 => dictionary!(Date32Type),
        DataType::Date64 => dictionary!(Date64Type),
        DataType::Time32(TimeUnit::Second) => dictionary!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => dictionary!(Time32MillisecondType),
        DataType::Time64(TimeUnit::Microsecond) => dictionary!(Time64MicrosecondType),
        DataType::Time64(TimeUnit::Nanosecond) => dictionary!(Time64NanosecondType),
        DataType::Timestamp(TimeUnit::Second, _) => dictionary!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            dictionary!(TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            dictionary!(TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            dictionary!(TimestampNanosecondType)
        }
        DataType::Duration(TimeUnit::Second) => dictionary!(DurationSecondType),
        DataType::Duration(TimeUnit::Millisecond) => dictionary!(DurationMillisecondType),
        DataType::Duration(TimeUnit::Microsecond) => dictionary!(DurationMicrosecondType),
        DataType::Duration(TimeUnit::Nanosecond) => dictionary!(DurationNanosecondType),
        DataType::Interval(IntervalUnit::YearMonth) => dictionary!(IntervalYearMonthType),
        DataType::Interval(IntervalUnit::DayTime) => dictionary!(IntervalDayTimeType),
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            dictionary!(IntervalMonthDayNanoType)
        }
        DataType::Decimal32(_, _) => dictionary!(Decimal32Type),
        DataType::Decimal64(_, _) => dictionary!(Decimal64Type),
        DataType::Decimal128(_, _) => dictionary!(Decimal128Type),
        DataType::Decimal256(_, _) => dictionary!(Decimal256Type),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        DictionaryArray, Int64Array, LargeStringArray, StringArray, StringViewArray,
        TimestampNanosecondArray,
    };
    use arrow::datatypes::Int32Type;

    fn strings(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn accumulator(
        data_type: DataType,
        sort_options: Option<SortOptions>,
        ignore_nulls: bool,
    ) -> DistinctArrayAggGroupsAccumulator {
        DistinctArrayAggGroupsAccumulator::try_new(&data_type, sort_options, ignore_nulls)
            .expect("supported type")
    }

    /// Renders a list array as one `Option<Vec<Option<String>>>` per row.
    fn lists(array: &ArrayRef) -> Vec<Option<Vec<Option<String>>>> {
        let array = array.as_list::<i32>();
        (0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    return None;
                }
                let values = array.value(row);
                let values = cast(values.as_ref(), &DataType::Utf8).expect("castable");
                Some(
                    values
                        .as_string::<i32>()
                        .iter()
                        .map(|value| value.map(str::to_string))
                        .collect(),
                )
            })
            .collect()
    }

    fn list(values: &[Option<&str>]) -> Option<Vec<Option<String>>> {
        Some(
            values
                .iter()
                .map(|value| value.map(str::to_string))
                .collect(),
        )
    }

    #[test]
    fn dedups_within_and_across_batches_in_first_seen_order() -> Result<()> {
        let mut acc = accumulator(DataType::Utf8, None, false);
        acc.update_batch(
            &[strings(&[Some("b"), Some("a"), Some("b"), None, Some("c")])],
            &[0, 0, 0, 1, 2],
            None,
            4,
        )?;
        acc.update_batch(
            &[strings(&[Some("a"), Some("c"), None, Some("c")])],
            &[0, 0, 1, 2],
            None,
            4,
        )?;
        assert_eq!(acc.num_groups(), 4);

        let result = acc.evaluate(EmitTo::All)?;
        assert_eq!(
            lists(&result),
            vec![
                list(&[Some("b"), Some("a"), Some("c")]),
                list(&[None]),
                list(&[Some("c")]),
                None,
            ]
        );
        assert!(acc.pairs.is_empty());
        assert_eq!(acc.dictionary.len(), 0);
        assert_eq!(acc.num_groups(), 0);
        Ok(())
    }

    #[test]
    fn ignore_nulls_and_filter_skip_rows() -> Result<()> {
        let mut acc = accumulator(DataType::Utf8, None, true);
        let filter = BooleanArray::from(vec![Some(true), Some(false), None, Some(true)]);
        acc.update_batch(
            &[strings(&[Some("a"), Some("b"), Some("c"), None])],
            &[0, 0, 0, 1],
            Some(&filter),
            2,
        )?;
        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("a")]), None]
        );
        Ok(())
    }

    #[test]
    fn order_by_sorts_each_group() -> Result<()> {
        let desc_nulls_first = SortOptions {
            descending: true,
            nulls_first: true,
        };
        let mut acc = accumulator(DataType::Utf8, Some(desc_nulls_first), false);
        acc.update_batch(
            &[strings(&[Some("a"), None, Some("c"), Some("b"), Some("a")])],
            &[0, 0, 0, 1, 1],
            None,
            2,
        )?;
        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![
                list(&[None, Some("c"), Some("a")]),
                list(&[Some("b"), Some("a")]),
            ]
        );
        Ok(())
    }

    #[test]
    fn state_merges_into_another_accumulator() -> Result<()> {
        let mut partial = accumulator(DataType::Utf8, None, false);
        partial.update_batch(
            &[strings(&[Some("a"), Some("b"), None])],
            &[0, 1, 1],
            None,
            2,
        )?;
        let state = partial.state(EmitTo::All)?;
        assert_eq!(state.len(), 1);

        let direct = accumulator(DataType::Utf8, None, false)
            .convert_to_state(&[strings(&[Some("b"), Some("z"), None])], None)?;

        let mut final_acc = accumulator(DataType::Utf8, None, false);
        // The partial state lands on groups 1 and 0, so group 0 gathers "b" and
        // null from the partial and "b" again from the converted row.
        final_acc.merge_batch(&state, &[1, 0], 2)?;
        final_acc.merge_batch(&direct, &[0, 1, 1], 2)?;
        assert_eq!(
            lists(&final_acc.evaluate(EmitTo::All)?),
            vec![
                list(&[Some("b"), None]),
                list(&[Some("a"), Some("z"), None]),
            ]
        );
        Ok(())
    }

    #[test]
    fn merge_batch_of_sliced_state_interns_only_the_slice() -> Result<()> {
        let mut partial = accumulator(DataType::Utf8, None, false);
        partial.update_batch(
            &[strings(&[Some("a"), Some("b"), Some("c"), Some("d")])],
            &[0, 1, 2, 3],
            None,
            4,
        )?;
        let state = partial.state(EmitTo::All)?;

        let mut final_acc = accumulator(DataType::Utf8, None, false);
        final_acc.merge_batch(&[state[0].slice(1, 2)], &[0, 1], 2)?;
        assert_eq!(final_acc.dictionary.len(), 2);
        assert_eq!(
            lists(&final_acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("b")]), list(&[Some("c")])]
        );
        Ok(())
    }

    #[test]
    fn convert_to_state_nulls_filtered_and_ignored_rows() -> Result<()> {
        let filter = BooleanArray::from(vec![true, false, true]);
        let input = strings(&[Some("a"), Some("b"), None]);

        let keep_nulls = accumulator(DataType::Utf8, None, false)
            .convert_to_state(&[Arc::clone(&input)], Some(&filter))?;
        assert_eq!(
            lists(&keep_nulls[0]),
            vec![list(&[Some("a")]), None, list(&[None])]
        );

        let ignore_nulls = accumulator(DataType::Utf8, None, true)
            .convert_to_state(&[input], Some(&filter))?;
        assert_eq!(
            lists(&ignore_nulls[0]),
            vec![list(&[Some("a")]), None, None]
        );
        Ok(())
    }

    #[test]
    fn emit_first_renumbers_groups_and_compacts_dictionary() -> Result<()> {
        let mut acc = accumulator(DataType::Utf8, None, false);
        acc.update_batch(
            &[strings(&[
                Some("a"),
                Some("b"),
                Some("b"),
                Some("c"),
                Some("d"),
            ])],
            &[0, 0, 1, 2, 2],
            None,
            3,
        )?;
        let size_before = acc.size();

        let first = acc.evaluate(EmitTo::First(2))?;
        assert_eq!(
            lists(&first),
            vec![list(&[Some("a"), Some("b")]), list(&[Some("b")])]
        );
        assert_eq!(acc.num_groups(), 1);
        // Only "c" and "d" are still referenced.
        assert_eq!(acc.dictionary.len(), 2);
        assert_eq!(acc.pairs.len(), 2);
        assert!(acc.size() < size_before);

        // The retained group is now group 0 and keeps accepting values.
        acc.update_batch(&[strings(&[Some("c"), Some("e")])], &[0, 1], None, 2)?;
        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("c"), Some("d")]), list(&[Some("e")])]
        );
        Ok(())
    }

    #[test]
    fn preserving_reads_leave_state_untouched() -> Result<()> {
        let mut acc = accumulator(DataType::Utf8, None, false);
        acc.update_batch(
            &[strings(&[Some("a"), Some("b"), Some("a")])],
            &[0, 2, 2],
            None,
            3,
        )?;

        let selection = GroupSelection::try_from_indices(&[2, 0, 2, 1], 3)?;
        let expected = vec![
            list(&[Some("b"), Some("a")]),
            list(&[Some("a")]),
            list(&[Some("b"), Some("a")]),
            None,
        ];
        for _ in 0..2 {
            assert_eq!(lists(&acc.evaluate_preserving(selection)?), expected);
        }
        let state = acc.state_preserving(selection)?;
        assert_eq!(lists(&state[0]), expected);

        let empty = acc.evaluate_preserving(GroupSelection::try_from_indices(&[], 3)?)?;
        assert!(empty.is_empty());
        assert_eq!(
            empty.data_type(),
            &DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true)))
        );

        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("a")]), None, list(&[Some("b"), Some("a")])]
        );
        Ok(())
    }

    #[test]
    fn dictionary_input_round_trips_to_dictionary_lists() -> Result<()> {
        let data_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let mut acc = accumulator(data_type.clone(), None, false);
        let values: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::from_iter([
            Some("x"),
            Some("y"),
            Some("x"),
        ]));
        acc.update_batch(&[Arc::clone(&values)], &[0, 0, 1], None, 2)?;

        let state = acc.state(EmitTo::All)?;
        assert_eq!(
            state[0].data_type(),
            &DataType::List(Arc::new(Field::new_list_field(data_type.clone(), true)))
        );
        let converted = acc.convert_to_state(&[values], None)?;
        assert_eq!(converted[0].data_type(), state[0].data_type());

        let mut final_acc = accumulator(data_type.clone(), None, false);
        final_acc.merge_batch(&state, &[0, 0], 1)?;
        final_acc.merge_batch(&converted, &[0, 0, 0], 1)?;
        let result = final_acc.evaluate(EmitTo::All)?;
        assert_eq!(
            result.data_type(),
            &DataType::List(Arc::new(Field::new_list_field(data_type, true)))
        );
        assert_eq!(lists(&result), vec![list(&[Some("x"), Some("y")])]);
        Ok(())
    }

    #[test]
    fn other_supported_types() -> Result<()> {
        let mut acc = accumulator(DataType::LargeUtf8, None, false);
        let values: ArrayRef = Arc::new(LargeStringArray::from(vec!["p", "q", "p"]));
        acc.update_batch(&[values], &[0, 0, 0], None, 1)?;
        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("p"), Some("q")])]
        );

        let mut acc = accumulator(DataType::Utf8View, None, false);
        let values: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("a long enough string to spill out of the view"),
            None,
            Some("a long enough string to spill out of the view"),
        ]));
        acc.update_batch(&[values], &[0, 0, 0], None, 1)?;
        let result = acc.evaluate(EmitTo::All)?;
        assert_eq!(
            result.as_list::<i32>().values().data_type(),
            &DataType::Utf8View
        );
        assert_eq!(
            lists(&result),
            vec![list(&[
                Some("a long enough string to spill out of the view"),
                None
            ])]
        );

        let mut acc = accumulator(DataType::Int64, None, false);
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(7), Some(7), None]));
        acc.update_batch(&[values], &[0, 1, 1], None, 2)?;
        assert_eq!(
            lists(&acc.evaluate(EmitTo::All)?),
            vec![list(&[Some("7")]), list(&[Some("7"), None])]
        );

        let timestamp = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));
        let mut acc = accumulator(timestamp.clone(), None, false);
        let values: ArrayRef =
            Arc::new(TimestampNanosecondArray::from(vec![1, 2, 1]).with_timezone("UTC"));
        acc.update_batch(&[values], &[0, 0, 0], None, 1)?;
        let result = acc.evaluate(EmitTo::All)?;
        assert_eq!(result.as_list::<i32>().values().data_type(), &timestamp);
        assert_eq!(result.as_list::<i32>().values().len(), 2);
        Ok(())
    }

    #[test]
    fn unsupported_types_are_rejected() {
        for data_type in [
            DataType::Float64,
            DataType::Boolean,
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
            DataType::FixedSizeBinary(4),
        ] {
            assert!(
                !DistinctArrayAggGroupsAccumulator::supports_type(&data_type),
                "{data_type}"
            );
            assert!(
                DistinctArrayAggGroupsAccumulator::try_new(&data_type, None, false)
                    .is_err()
            );
        }
        assert!(DistinctArrayAggGroupsAccumulator::supports_type(
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8View))
        ));
    }

    #[test]
    fn size_tracks_growth_and_release() -> Result<()> {
        let mut acc = accumulator(DataType::Utf8, None, false);
        let empty = acc.size();
        let values: Vec<String> = (0..10_000).map(|i| format!("value{i}")).collect();
        let values =
            strings(&values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>());
        let groups: Vec<usize> = (0..10_000).map(|i| i % 100).collect();
        acc.update_batch(&[values], &groups, None, 100)?;
        let full = acc.size();
        assert!(
            full > empty + 10_000 * size_of::<u64>(),
            "{full} vs {empty}"
        );
        acc.evaluate(EmitTo::All)?;
        assert!(acc.size() < full / 2, "{} vs {full}", acc.size());
        Ok(())
    }
}
