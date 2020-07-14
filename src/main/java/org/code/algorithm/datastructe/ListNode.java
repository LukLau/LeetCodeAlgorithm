package org.code.algorithm.datastructe;

import lombok.Data;

/**
 * @author luk
 * @date 2020/6/25
 */
public class ListNode {
    public int val;
    public ListNode next;

    ListNode() {
    }

    public ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
