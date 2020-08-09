package org.code.algorithm.swordoffer;

import java.util.Stack;

/**
 *
 * @author luk
 * @date 2020/7/26
 */
public class MinStack {

    private final Stack<Integer> stack = new Stack<>();

    private final Stack<Integer> minStack = new Stack<>();

    public void push(int node) {
        stack.push(node);

        if (minStack.isEmpty()) {
            minStack.push(node);
        } else {
            Integer peek = minStack.peek();
            if (node <= peek) {
                minStack.push(node);
            }
        }

    }

    public void pop() {
        Integer pop = stack.pop();
        if (pop.equals(minStack.peek())) {
            minStack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int min() {
        return minStack.peek();
    }
}
