package org.code.algorithm.datastructe;

import jdk.nashorn.internal.ir.ReturnNode;

import java.util.Iterator;
import java.util.List;

/**
 * @author luk
 * @date 2020/11/17
 */
public class ZigzagIterator {

    private final Iterator<Integer> iterator1;

    private final Iterator<Integer> iterator2;

    private volatile boolean leftToRight = true;
    /**
     * @param v1: A 1d vector
     * @param v2: A 1d vector
     */
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        iterator1 = v1.iterator();
        iterator2 = v2.iterator();


    }

    /**
     * @return: An integer
     */
    public int next() {
        if (!iterator1.hasNext()) {
            return iterator2.next();
        }
        if (!iterator2.hasNext()) {
            return iterator1.next();
        }
        boolean tmp = leftToRight;
        leftToRight = !leftToRight;
        if (tmp) {
            return iterator1.next();
        } else {
            return iterator2.next();

        }
    }

    /**
     * @return: True if has next
     */
    public boolean hasNext() {
        // write your code here
        return iterator1.hasNext() || iterator2.hasNext();
    }

}
