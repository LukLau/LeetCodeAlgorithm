package org.code.algorithm.swordoffer;

import java.util.Stack;

/**
 * @author luk
 * @date 2020/7/25
 */
public class Solution {


    private final Stack<Integer> leftSide = new Stack<>();
    private final Stack<Integer> rightSide = new Stack<>();

    public void push(int node) {
        leftSide.push(node);
    }

    public int pop() {
        if (!rightSide.isEmpty()) {
            return rightSide.pop();
        }
        while (!leftSide.isEmpty()) {
            Integer pop = leftSide.pop();

            rightSide.push(pop);
        }
        return rightSide.pop();
    }

}

