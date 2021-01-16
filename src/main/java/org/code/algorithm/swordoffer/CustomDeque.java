package org.code.algorithm.swordoffer;

import java.util.Stack;

/**
 * 两个栈实现队列
 *
 * @author luk
 * @date 2020/11/25
 */
public class CustomDeque {

    public void push(int node) {
        head.push(node);
    }

    public int pop() {
        if (!tail.isEmpty()) {
            return tail.pop();
        }
        while (!head.isEmpty()) {
            tail.push(head.pop());
        }
        return tail.pop();
    }

    private final Stack<Integer> head = new Stack<>();
    private final Stack<Integer> tail = new Stack<>();

}
