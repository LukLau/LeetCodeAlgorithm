package org.code.algorithm.datastructe;

/**
 * @author dora
 * @date 2020/8/12
 */
public class Node {

    public int val;
    public int key;
    public Node left;
    public Node right;
    public Node next;
    public Node random;
    public Node prev;

    public Node() {
    }

    public Node( int key, int val) {
        this.val = val;
        this.key = key;
    }

    public Node(int val) {
        this.val = val;
    }

    public Node(int val, Node left, Node right, Node next) {
        this.val = val;
        this.left = left;
        this.right = right;
        this.next = next;
    }
}
