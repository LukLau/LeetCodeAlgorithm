package org.code.algorithm.datastructe;

import java.util.Iterator;

/**
 * 284. Peeking Iterator
 *
 * @author luk
 * @date 2020/11/17
 */
public class PeekingIterator implements Iterator<Integer> {

    private final Iterator<Integer> iterator;

    private Integer currentValue = null;

    public PeekingIterator(Iterator<Integer> iterator) {
        this.iterator = iterator;
        if (iterator != null) {
            currentValue = iterator.next();
        }
        // initialize any member here.
    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        return currentValue;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        Integer tmp = currentValue;
        if (iterator.hasNext()) {
            currentValue = iterator.next();
        } else {
            currentValue = null;
        }
        return tmp;
    }

    @Override
    public boolean hasNext() {
        return currentValue != null;
    }

}
