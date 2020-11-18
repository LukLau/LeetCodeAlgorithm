package org.code.algorithm.datastructe;

import java.util.Iterator;
import java.util.List;

/**
 * @author luk
 * @date 2020/11/17
 */
public class ZigzagIterator {


    private Iterator<Integer> iterator1;

    private Iterator<Integer> iterator2;

    private boolean leftToRight = true;

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
        int result;

        if (leftToRight) {
            result = iterator1.next();
        } else {
            result = iterator2.next();
        }
        leftToRight = !leftToRight;
        // write your code here
        return result;
    }

    /**
     * @return: True if has next
     */
    public boolean hasNext() {
        // write your code here
        return iterator1.hasNext() || iterator2.hasNext();
    }

}
