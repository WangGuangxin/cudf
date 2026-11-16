/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Options for reading a parquet file
 */
public class ParquetOptions extends ColumnFilterOptions {

  public static ParquetOptions DEFAULT = new ParquetOptions(new Builder());

  private final DType unit;
  private final boolean[] readBinaryAsString;
  private final boolean serializeColumnChunkReads;

  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
    serializeColumnChunkReads = builder.serializeColumnChunkReads;
    readBinaryAsString = new boolean[builder.binaryAsStringColumns.size()];
    for (int i = 0 ; i < builder.binaryAsStringColumns.size() ; i++) {
      readBinaryAsString[i] = builder.binaryAsStringColumns.get(i);
    }
  }

  DType timeUnit() {
    return unit;
  }

  boolean[] getReadBinaryAsString() {
    return readBinaryAsString;
  }

  /**
   * Whether column chunk reads are serialized until completion across concurrent readers.
   * @see Builder#withSerializeColumnChunkReads(boolean)
   */
  public boolean isSerializeColumnChunkReads() {
    return serializeColumnChunkReads;
  }

  public static ParquetOptions.Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private DType unit = DType.EMPTY;
    private boolean serializeColumnChunkReads = false;
    final List<Boolean> binaryAsStringColumns = new ArrayList<>();

    /**
     * Set whether column chunk reads are serialized until completion across concurrent readers.
     * When enabled, each reader on a GPU transfers its column chunk data and waits for the
     * transfers to complete before another reader with this option enabled starts its own. With
     * many concurrent readers this lets earlier readers start decoding while later readers are
     * still transferring, instead of all readers' transfers completing together. It can lower
     * throughput for data sources that rely on concurrent device reads, such as GPUDirect
     * Storage. Disabled by default.
     * @param serialize whether to serialize column chunk reads until completion
     * @return builder for chaining
     */
    public Builder withSerializeColumnChunkReads(boolean serialize) {
      this.serializeColumnChunkReads = serialize;
      return this;
    }

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(DType unit) {
      assert unit.isTimestampType();
      this.unit = unit;
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(String... names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.length ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    /**
     * Include this column.
     * @param name the name of the column
     * @param isBinary whether this column is to be read in as binary
     */
    public Builder includeColumn(String name, boolean isBinary) {
      includeColumnNames.add(name);
      binaryAsStringColumns.add(!isBinary);
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(Collection<String> names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.size() ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    public ParquetOptions build() {
      return new ParquetOptions(this);
    }
  }
}
