package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.Node;

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


}
