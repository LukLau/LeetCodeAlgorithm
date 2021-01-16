package org.code.algorithm.datastructe;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author luk
 * @date 2020/10/29
 */
public class Vector2D implements Iterator<Integer> {

    private Iterator<List<Integer>> listIterator;

    private Iterator<Integer> iterator;


    public Vector2D(List<List<Integer>> vec2d) {
        listIterator = vec2d.iterator();

        // Initialize your data structure here
    }

    public static void main(String[] args) {
        List<List<Integer>> params = new ArrayList<>();
//                Arrays.asList(new ArrayList<>(), Arrays.asList(1, 2), new ArrayList<>());
        Vector2D vector2D = new Vector2D(params);
    }

    @Override
    public Integer next() {
        // Write your code here
//        Integer result = null;
//        while (result == null) {
//            if (this.hasNext()) {
//                result = iterator.next();
//            } else {
//                return null;
//            }
//        }
//        return result;
        if (hasNext()) {
            return iterator.next();
        }
        return null;
    }

    @Override
    public boolean hasNext() {
        if (iterator == null && !listIterator.hasNext()) {
            return false;
        }
        // Write your code here
        while (iterator == null || (!iterator.hasNext() && listIterator.hasNext())) {

            List<Integer> tmp = listIterator.next();
            iterator = tmp.iterator();
        }
        return iterator.hasNext();
    }

    @Override
    public void remove() {
    }
}
