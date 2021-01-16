package org.code.algorithm.swordoffer;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 *
 * @author dora
 * @date 2020/8/5
 */
public class MediaSolution {
    private final PriorityQueue<Integer> small = new PriorityQueue<>(Comparator.reverseOrder());

    private final PriorityQueue<Integer> big = new PriorityQueue<>();

    public void Insert(Integer num) {
        small.offer(num);

        big.offer(small.poll());

        if (big.size() > small.size()) {
            small.offer(big.poll());
        }
    }

    public Double GetMedian() {
        if (big.size() < small.size()) {
            return small.peek() / 1.0;
        }
        return (small.peek() + big.peek()) / 2.0;
    }

}
