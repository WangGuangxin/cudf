/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.CompiledExpression;

/**
 * Hash table over the right (build) side of a mixed (equality + AST condition) left semi/anti
 * join that can be probed with many left tables. Gives the same results as
 * {@link Table#mixedLeftSemiJoinGatherMap} / {@link Table#mixedLeftAntiJoinGatherMap} without
 * rebuilding the hash table per call.
 *
 * The right tables are referenced, not copied: their device memory must stay alive and unchanged
 * until this object is closed.
 */
public final class MixedSemiJoinHashTable implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private final long rightRows;

  public MixedSemiJoinHashTable(Table rightKeys, Table rightConditional,
                                NullEquality nullEquality) {
    this.rightRows = rightKeys.getRowCount();
    this.nativeHandle = create(rightKeys.getNativeView(), rightConditional.getNativeView(),
        nullEquality == NullEquality.EQUAL);
  }

  public long getRightRowCount() {
    return rightRows;
  }

  public GatherMap leftSemiJoinGatherMap(Table leftKeys, Table leftConditional,
                                         CompiledExpression condition) {
    return toGatherMap(probe(handle(), leftKeys.getNativeView(), leftConditional.getNativeView(),
        condition.getNativeHandle(), true));
  }

  public GatherMap leftAntiJoinGatherMap(Table leftKeys, Table leftConditional,
                                         CompiledExpression condition) {
    return toGatherMap(probe(handle(), leftKeys.getNativeView(), leftConditional.getNativeView(),
        condition.getNativeHandle(), false));
  }

  private long handle() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("MixedSemiJoinHashTable is closed");
    }
    return nativeHandle;
  }

  private static GatherMap toGatherMap(long[] data) {
    return new GatherMap(DeviceMemoryBuffer.fromRmm(data[1], data[0], data[2]));
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0) {
      destroy(nativeHandle);
      nativeHandle = 0;
    }
  }

  private static native long create(long rightKeysTable, long rightConditionTable,
                                    boolean compareNullsEqual);

  private static native long[] probe(long handle, long leftKeysTable, long leftConditionTable,
                                     long condition, boolean semi);

  private static native void destroy(long handle);
}
