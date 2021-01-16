package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;

/**
 * @author luk
 * @date 2020/12/21
 */
public class ListSolution {

    /**
     * 86. Partition List
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }
        ListNode merge1 = new ListNode(0);

        ListNode merge2 = new ListNode(0);

        ListNode small = merge1;

        ListNode big = merge2;
        while (head != null) {
            if (head.val <= x) {
                small.next = head;
                small = small.next;
            } else {
                big.next = head;
                big = big.next;
            }
            head = head.next;
        }

        big.next = null;

        small.next = merge2.next;

        merge2.next = null;

        return merge1.next;

    }
}
