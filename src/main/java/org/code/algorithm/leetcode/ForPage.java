package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.Point;

import java.util.*;

/**
 * @author luk
 * @date 2020/11/19
 */
public class ForPage {


    public static void main(String[] args) {
        ForPage page = new ForPage();
        int[][] matrix = new int[][]{{1, 1}, {0, 1}, {3, 3}, {3, 4}};
        Point[] points = new Point[matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            points[i] = new Point(matrix[i][0], matrix[i][1]);
        }
        page.numIslands2V2(4, 5, points);
    }

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
     * todo
     *
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

    public List<Integer> numIslands2V2(int n, int m, Point[] operators) {
        if (operators == null || operators.length == 0) {
            return new ArrayList<>();
        }
        int[][] matrix = new int[n][m];
        List<Integer> result = new ArrayList<>();
//        Map<Point, Integer> map = new HashMap<>();
        int[][] params = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        for (Point operator : operators) {
            int x = operator.x;
            int y = operator.y;
            matrix[x][y] = 1;
            Integer count = intervalNumIslands2(x, y, params, matrix);
            int previousCount = result.isEmpty() ? 0 : result.get(result.size() - 1);
            if (count == 0) {
                result.add(previousCount);
            } else {
                result.add(previousCount + 1);
            }
        }
        return result;
    }

    private Integer intervalNumIslands2(int x, int y, int[][] direction, int[][] matrix) {
//        map.put(new Point(x, y), 1);
        for (int i = 0; i < direction.length; i++) {
            int neighborX = direction[i][0] + x;
            int neighborY = direction[i][1] + y;
            if (neighborX < 0 || neighborX >= matrix.length || neighborY < 0 || neighborY >= matrix[x].length) {
                continue;
            }
            if (matrix[neighborX][neighborY] == 1) {
                return 0;
            }
//            Point point = new Point(neighborX, neighborY);
//            if (map.containsKey(point)) {
//                return map.get(point);
//            }

        }
        return 1;
    }
}
