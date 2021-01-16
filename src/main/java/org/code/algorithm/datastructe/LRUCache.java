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
    private final Node head;

    private final Node tail;

    private final int capacity;

    private Map<Integer, Node> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;

        head = new Node();

        tail = new Node();

        head.next = tail;
        tail.prev = head;

        map = new HashMap<>();
    }


//    public static void main(String[] args) {
//        LRUCache cache = new LRUCache(2);
//        cache.put(1, 1);
//        cache.put(2, 2);
//        cache.get(1);
//        cache.put(3, 3);
//        cache.get(2);
//        cache.put(4, 4);
//        cache.get(1);
//        cache.get(3);
//        cache.get(4);
//    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        changeNode(node);
        return node.val;
    }

    private void changeNode(Node node) {
        removeNode(node);
        addFirst(node);
    }


    private void removeNode(Node node) {
        Node prev = node.prev;

        Node next = node.next;


        prev.next = next;

        next.prev = prev;

        node.prev = null;

        node.next = null;
    }

    private void addFirst(Node node) {
        Node next = head.next;

        head.next = node;

        node.prev = head;

        node.next = next;

        next.prev = node;
    }

    private Node popTailNode() {
        Node lastNode = tail.prev;

        lastNode.prev.next = tail;

        tail.prev = lastNode.prev;

        lastNode.prev = null;

        lastNode.next = null;

        return lastNode;
    }

    public void put(int key, int val) {
        Node node = map.get(key);
        if (node != null) {
            node.val = val;
            changeNode(node);
            return;
        }
        node = new Node(key, val);
        map.put(key, node);
        addFirst(node);
        if (map.size() > this.capacity) {

            Node tailNode = popTailNode();
            map.remove(tailNode.key);
        }
    }

}
