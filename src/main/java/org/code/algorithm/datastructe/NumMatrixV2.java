package org.code.algorithm.datastructe;

/**
 * todo
 * Binary Indexed Tree
 * 	308
 * Range Sum Query 2D - Mutable
 * @author luk
 * @date 2020/11/23
 */
public class NumMatrixV2 {
    private final int[][] matrix;

    public NumMatrixV2(int[][] matrix) {
        this.matrix = matrix;

    }

    public void update(int row, int col, int val) {
        matrix[row][col] = val;
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
