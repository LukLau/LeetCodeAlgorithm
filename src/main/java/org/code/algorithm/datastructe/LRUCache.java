package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * todo
 * @author dora
 * @date 2020/8/17
 */
public class LRUCache {

    private Map<Integer, Node> map = new HashMap<>();

    private Node head;

    private Node tail;

    private int capacity;


    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node();
        tail = new Node();

        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }

        return -1;
    }

    public void put(int key, int value) {

    }

}
