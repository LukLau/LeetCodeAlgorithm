package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.Point;

import java.util.*;

/**
 * @author luk
 * @date 2020/11/19
 */
public class ForPage {


    /**
     * todo
     * 301. Remove Invalid Parentheses
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        if (s == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        LinkedList<String> deque = new LinkedList<>();
        deque.offer(s);
        while (!deque.isEmpty()) {
            String poll = deque.poll();
            if (intervalInvalid(poll)) {
                result.add(poll);
            }
            if (!result.isEmpty()) {
                continue;
            }
            int len = poll.length();
            for (int i = 0; i < len; i++) {
                char word = s.charAt(i);
                if (word != '(' && word != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);

                if (!visited.contains(tmp)) {
                    visited.add(tmp);
                    deque.offer(tmp);
                }
            }
        }
        if (result.isEmpty()) {
            result.add("");
        }
        return result;
    }

    private boolean intervalInvalid(String poll) {
        if (poll == null || poll.isEmpty()) {
            return false;
        }
        int count = 0;
        char[] words = poll.toCharArray();
        for (char word : words) {
            if (word != '(' && word != ')') {
                continue;
            }
            if (word == '(') {
                count++;
            } else {
                if (count == 0) {
                    return false;
                }
                count--;
            }
        }
        return count == 0;
    }


    /**
     * 302 Smallest Rectangle Enclosing Black Pixels
     *
     * @param image: a binary matrix with '0' and '1'
     * @param x:     the location of one of the black pixels
     * @param y:     the location of one of the black pixels
     * @return: an integer
     */
    public int minArea(char[][] image, int x, int y) {
        // write your code here
        int row = image.length;
        int column = image[0].length;
        int left = column;
        int right = 0;
        int up = row;
        int bottom = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (image[i][j] == '1') {
                    left = Math.min(left, j);
                    right = Math.max(right, j);
                    up = Math.min(up, i);
                    bottom = Math.max(bottom, j);
                }
            }
        }
        return (right - left + 1) * (bottom - up + 1);
    }


    /**
     * todo
     *
     * @param image
     * @param x
     * @param y
     * @return
     */
    public int minAreaV2(char[][] image, int x, int y) {
        int row = image.length;
        int column = image[0].length;
        int up = getIndex(image, true, 0, x, 0, column, true);
        int down = getIndex(image, true, x + 1, row, 0, column, false);
        int left = getIndex(image, false, 0, y, up, down, true);
        int right = getIndex(image, false, y + 1, column, up, down, false);
        return (right - left) * (down - up);
    }

    private int getIndex(char[][] image, boolean vertx, int i, int j, int low, int high, boolean opt) {
        while (i < j) {
            int k = low, mid = (i + j) / 2;
            while (k < high && (vertx ? image[mid][k] : image[k][mid]) == '0') {
                ++k;
            }
            if (k < high == opt) {
                j = mid;
            } else {
                i = mid + 1;
            }
        }
        return i;
    }


    /**
     * @param n:         An integer
     * @param m:         An integer
     * @param operators: an array of point
     * @return: an integer array
     */
    public List<Integer> numIslands2(int n, int m, Point[] operators) {
        // write your code here
        int[][] matrix = new int[n][m];
        LinkedList<Point> deque = new LinkedList<>();
        for (Point operator : operators) {
            deque.offer(operator);
        }
        int[][] params = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        List<Integer> result = new ArrayList<>();
        while (!deque.isEmpty()) {
            Point poll = deque.poll();
            int x = poll.x;
            int y = poll.y;
            int count = 0;
            if (matrix[x][y] == -1) {
                result.add(count);
                continue;
            }
            count++;
            for (int[] param : params) {
                x = x + param[0];
                y = y + param[1];
                if (x < 0 || x >= matrix.length || y < 0 || y >= matrix[x].length || matrix[x][y] == -1) {
                    continue;
                }
                matrix[x][y] = -1;
                count++;
            }
        }
        return result;
    }

    private int intervalNumIslands2(int x, int y, int[][] matrix, int count) {
        return -1;
    }


    public static void main(String[] args) {
        ForPage page = new ForPage();
    }

}
