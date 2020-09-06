package org.code.algorithm.datastructe;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.Stack;

/**
 * @author luk
 * @date 2020/9/6
 */
public class BSTIterator {


    private Stack<TreeNode> stack = new Stack<>();

    private Iterator<Integer> tmp = null;

    private LinkedList<Integer> iterator = new LinkedList<>();

    public BSTIterator(TreeNode root) {
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            iterator.add(p.val);

            p = p.right;
        }
        tmp = iterator.iterator();
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return tmp.next();
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return tmp.hasNext();
    }
}
