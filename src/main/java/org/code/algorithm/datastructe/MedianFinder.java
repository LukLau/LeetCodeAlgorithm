package org.code.algorithm.datastructe;

import java.util.PriorityQueue;

/**
 * 295. Find Median from Data Stream
 *
 * @author luk
 * @date 2020/11/19
 */
public class MedianFinder {


    private PriorityQueue<Integer> small = new PriorityQueue<>((o1, o2) -> o2 - o1);

    private PriorityQueue<Integer> big = new PriorityQueue<>();

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {

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
            return small.peek();
        }
        return ((double) small.peek() + (double) big.peek()) / 2;
    }


    public static void main(String[] args) {
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);
        System.out.println(medianFinder.findMedian());
        medianFinder.addNum(2);
        System.out.println(medianFinder.findMedian());
        medianFinder.addNum(3);
        System.out.println(medianFinder.findMedian());
        medianFinder.addNum(4);
        System.out.println(medianFinder.findMedian());
    }

}
