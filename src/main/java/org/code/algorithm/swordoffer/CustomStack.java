package org.code.algorithm.swordoffer;

import java.util.Stack;

/**
 * 最小栈
 *
 * @author luk
 * @date 2020/11/25
 */
public class CustomStack {
    private final Stack<Integer> stack = new Stack<>();

    private final Stack<Integer> minStack = new Stack<>();

    public void push(int node) {
        stack.push(node);
        if (minStack.isEmpty() || minStack.peek() >= node) {
            minStack.push(node);
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
