package org.code.algorithm.datastructe;

import java.util.Iterator;

/**
 * 284. Peeking Iterator
 *
 * @author luk
 * @date 2020/11/17
 */
public class PeekingIterator implements Iterator<Integer> {

    private Iterator<Integer> iterator;
    private Integer currentVal = null;

    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;
        if (iterator.hasNext()) {
            currentVal = iterator.next();
        }
    }

    @Override
    public boolean hasNext() {
        return currentVal != null;
    }

    @Override
    public Integer next() {
        int result = currentVal;
        currentVal = iterator.hasNext() ? iterator.next() : null;
        return result;
    }


    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        return currentVal;
    }

}
