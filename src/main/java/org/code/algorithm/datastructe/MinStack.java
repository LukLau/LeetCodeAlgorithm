package org.code.algorithm.datastructe;

import java.util.Stack;

/**
 * @author dora
 * @date 2020/8/26
 */
public class MinStack {


    private final Stack<Integer> stack;

    private final Stack<Integer> minStack;

    /**
     * initialize your data structure here.
     */
    public MinStack() {

        stack = new Stack<>();

        minStack = new Stack<>();

    }

    public void push(int x) {
        stack.push(x);

        if (minStack.isEmpty() || x <= minStack.peek()) {
            minStack.push(x);
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

    public int getMin() {
        return minStack.peek();
    }
}
