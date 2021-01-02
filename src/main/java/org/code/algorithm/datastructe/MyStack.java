package org.code.algorithm.datastructe;

import java.util.LinkedList;

/**
 * 225. Implement Stack using Queues
 *
 * @author luk
 */
public class MyStack {
    private final LinkedList<Integer> queue;

    public MyStack() {
        this.queue = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        queue.offer(x);
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        int size = queue.size();
        for (int i = 0; i < size - 1; i++) {
            Integer pop = queue.pop();
            queue.offer(pop);
        }
        return queue.pollFirst();

    }

    /** Get the top element. */
    public int top() {
        int size = queue.size();

        return queue.get(size -1);
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue.isEmpty();
    }
}
