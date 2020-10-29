package org.code.algorithm.datastructe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * @author luk
 * @date 2020/10/29
 */
public class Vector2D implements Iterator<Integer> {

    private Iterator<List<Integer>> iterator;

    private Iterator<Integer> secondIterator;

    public Vector2D(List<List<Integer>> vec2d) {
        if (!vec2d.isEmpty()) {
            // Initialize your data structure here
            this.iterator = vec2d.iterator();
            this.secondIterator = this.iterator.next().iterator();
        }
    }

    @Override
    public Integer next() {
        if (iterator == null) {
            return null;
        }
        while (iterator != null && secondIterator != null) {
            if (secondIterator.hasNext()) {
                return secondIterator.next();
            }
            if (!iterator.hasNext()) {
                return null;
            }
            secondIterator = iterator.next().iterator();
        }
        return null;
    }


    @Override
    public boolean hasNext() {
        if (iterator == null) {
            return false;
        }
        if (secondIterator.hasNext()) {
            return true;
        }
        if (!iterator.hasNext()) {
            return false;
        }
        secondIterator = iterator.next().iterator();

        return secondIterator.hasNext();
        // Write your code here
    }

    @Override
    public void remove() {
    }

    public static void main(String[] args) {
        Vector2D vector2D = new Vector2D(Arrays.asList(new ArrayList<>(), new ArrayList<>(), Arrays.asList(1, 2, 3), new ArrayList<>(), Arrays.asList(4)));
        System.out.println(vector2D.next());
        System.out.println(vector2D.next());
        System.out.println(vector2D.next());
        System.out.println(vector2D.next());
        System.out.println(vector2D.next());
        System.out.println(vector2D.next());


    }
}
