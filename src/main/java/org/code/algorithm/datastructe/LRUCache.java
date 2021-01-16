package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * LRU last recently used
 *
 * @author dora
 * @date 2020/8/17
 */
public class LRUCache {
    private Node head;

    private Node tail;

    private int capacity;

    private Map<Integer, Node> map;


    public LRUCache(int capacity) {

        head = new Node();
        tail = new Node();

        head.next = tail;
        tail.prev = head;

        map = new HashMap<>();
        this.capacity = capacity;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        intervalLru(node);
        return node.val;
    }

    private void intervalLru(Node node) {
        removeNode(node);

        addToFirst(node);
    }

    private void removeNode(Node node) {
        Node prevNode = node.prev;

        Node nextNode = node.next;

        prevNode.next = nextNode;

        nextNode.prev = prevNode;

        node.next = null;

        node.prev = null;
    }

    private void addToFirst(Node node) {
        Node previousFirst = head.next;

        head.next = node;

        node.prev = head;

        node.next = previousFirst;

        previousFirst.prev = node;
    }

    private Node popTailNode() {
        Node lastNode = tail.prev;

        lastNode.prev.next = tail;

        tail.prev = lastNode.prev;

        lastNode.prev = null;

        lastNode.next = null;

        return lastNode;
    }

    public void put(int key, int value) {
        Node node = map.get(key);

        if (node != null) {
            node.val = value;
            intervalLru(node);
            return;
        }
        if (map.size() == capacity) {
            Node tailNode = popTailNode();

            map.remove(tailNode.key);
        }
        node = new Node(key, value);

        map.put(key, node);

        addToFirst(node);
    }

    public static void main(String[] args) {
        LRUCache cache = new LRUCache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        cache.get(1);
        cache.put(3, 3);
        cache.get(2);
        cache.put(4, 4);
        cache.get(1);
        cache.get(3);
        cache.get(4);
    }

}
