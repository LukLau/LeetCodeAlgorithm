package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.Point;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 图系列问题
 *
 * @author luk
 * @date 2020/10/15
 */
public class GraphSolution {


    /**
     * 133. Clone Graph
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }


    /**
     * 等价于判断图中是否有循环
     * todo
     * 207. Course Schedule
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (prerequisites == null || prerequisites.length == 0) {
            return true;
        }

        return false;
    }

    /**
     * 210. Course Schedule II
     * todo
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        return null;

    }


    /**
     * todo
     * <p>
     * 261
     * Graph Valid Tree
     * 判断图是否是一个有效树
     *
     * @param n:     An integer
     * @param edges: a list of undirected edges
     * @return: true if it's a valid tree, or false
     */
    public boolean validTree(int n, int[][] edges) {
        if (edges == null) {
            return false;
        }
        if (n == 1 && edges.length == 0) {
            return true;
        }
        if (edges.length != n - 1) {
            return false;
        }
        return true;
    }


    /**
     * 305
     * Number of Islands II
     * todo 连通图算法
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


    /**
     * todo 310. Minimum Height Trees
     *
     * @param n
     * @param edges
     * @return
     */
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        return null;
    }


}
