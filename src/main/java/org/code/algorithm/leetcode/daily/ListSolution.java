package org.code.algorithm.leetcode.daily;

import org.code.algorithm.datastructe.Node;

/**
 * 链表问题
 *
 * @author luk
 * @date 2020/10/15
 */
public class ListSolution {


    /**
     * 138. Copy List with Random Pointer
     *
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        Node currentNode = head;
        while (currentNode != null) {
            Node node = new Node(currentNode.val);

            node.next = currentNode.next;

            currentNode.next = node;

            currentNode = node.next;
        }

        currentNode = head;
        while (currentNode != null) {
            Node random = currentNode.random;

            if (random != null) {
                currentNode.next.random = random.next;
            }
            currentNode = currentNode.next.next;
        }
        currentNode = head;
        Node copyHead = head.next;
        while (currentNode.next != null) {
            Node next = currentNode.next;
            currentNode.next = next.next;
            currentNode = next;
        }
        return copyHead;

    }

}
