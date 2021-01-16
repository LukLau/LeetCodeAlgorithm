package org.code.algorithm.datastructe;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * 295. Find Median from Data Stream
 *
 * @author luk
 * @date 2020/11/19
 */
public class MedianFinder {

    private final PriorityQueue<Integer> small;

    private PriorityQueue<Integer> big;

    public MedianFinder() {
        small = new PriorityQueue<>(Comparator.reverseOrder());
        big = new PriorityQueue<>();
    }

    public void addNum(int num) {
        small.offer(num);
        big.offer(small.poll());
        if (big.size() > small.size()) {
            small.offer(big.poll());
        }

    }

    public double findMedian() {
        if (small.size() > big.size()) {
            return small.peek() * 1.0;
        }
        return (small.peek() + big.peek()) / 2.0;
    }

}
