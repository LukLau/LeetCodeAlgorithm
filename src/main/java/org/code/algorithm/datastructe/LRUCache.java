package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * todo
 *
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

    public static void main(String[] args) {
        LRUCache cache = new LRUCache(1);

        cache.put(2, 1);

        cache.get(2);
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        removeNode(node);

        moveToFirst(node);

        return node.val;
    }

    private void removeNode(Node node) {
        Node prev = node.prev;

        Node next = node.next;

        prev.next = next;

        next.prev = prev;

        node.next = null;

        node.prev = null;
    }

    private void moveToFirst(Node node) {
        Node next = head.next;

        head.next = node;

        node.prev = head;

        node.next = next;

        next.prev = node;
    }

    private Node getTailNode() {
        Node tmp = tail.prev;

        removeNode(tmp);

        return tmp;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.val = value;
            removeNode(node);
            moveToFirst(node);
            return;
        }
        Node newNode = new Node(value, key);

        if (map.size() == capacity) {

            Node tailNode = getTailNode();

            map.remove(tailNode.key);
        }
        map.put(key, newNode);
        moveToFirst(newNode);

    }

}
