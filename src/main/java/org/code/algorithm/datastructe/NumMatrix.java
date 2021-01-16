package org.code.algorithm.datastructe;

/**
 * Binary Indexed Tree
 * 线段树问题
 * todo 304 Range Sum Query 2D - Immutable
 *
 * @author luk
 * @date 2020/11/20
 */
public class NumMatrix {

    private final int[][] matrix;

    public NumMatrix(int[][] matrix) {
        this.matrix = matrix;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        int result = 0;
        for (int i = row1; i <= row2; i++) {
            for (int j = col1; j <= col2; j++) {
                result += this.matrix[i][j];
            }
        }
        return result;
    }

}
