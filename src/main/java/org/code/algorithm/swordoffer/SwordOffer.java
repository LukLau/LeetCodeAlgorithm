package org.code.algorithm.swordoffer;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 * @author dora
 * @date 2020/7/25
 */
public class SwordOffer {

    public static void main(String[] args) {
        SwordOffer swordOffer = new SwordOffer();
        swordOffer.replaceSpace(new StringBuffer(" "));
    }

    /**
     * 二维数组中的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int value = array[i][j];

            if (value == target) {
                return true;
            } else if (value < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return "";
        }
        String word = str.toString();

        int blankWord = 0;
        for (char tmp : word.toCharArray()) {
            if (tmp == ' ') {
                blankWord++;
            }
        }
        StringBuilder builder = new StringBuilder();

        String[] words = word.split(" ");

        for (String s : words) {
            builder.append(s);
            if (blankWord-- > 0) {
                builder.append("%20");
            }
        }
        while (blankWord-- > 0) {
            builder.append("%20");
        }
        return builder.toString();
    }


    /**
     * 从尾到头打印链表
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> linkedList = new LinkedList<>();
        while (listNode != null) {
            linkedList.addFirst(listNode.val);
            listNode = listNode.next;
        }
        return new ArrayList<>(linkedList);
    }


    /**
     * 前序 中序重建二叉树
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        return intervalConstruct(pre, 0, in, 0, in.length - 1);
    }

    private TreeNode intervalConstruct(int[] pre, int preStart, int[] in, int inStart, int inEnd) {
        if (preStart >= pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = intervalConstruct(pre, preStart + 1, in, inStart, index - 1);
        root.right = intervalConstruct(pre, preStart + index - inStart + 1, in, index + 1, inEnd);
        return root;
    }


    /**
     * 旋转数组的最小值
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            if (array[left] < array[right]) {
                return array[left];
            }
            int mid = left + (right - left) / 2;

            if (array[left] <= array[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }


    public int minNumberInRotateArrayV2(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] <= array[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return array[left];
    }


    /**
     * 斐波那契
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        int sum = 0;
        int sum1 = 0;
        int sum2 = 1;
        for (int i = 2; i <= n; i++) {
            sum = sum1 + sum2;
            sum1 = sum2;
            sum2 = sum;
        }
        return sum;
    }

    /**
     * 跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target <= 0) {
            return 0;
        }
        if (target <= 2) {
            return target;
        }
        int result = 0;
        int jump1 = 1;
        int jump2 = 2;
        for (int i = 3; i <= target; i++) {
            result = jump1 + jump2;
            jump1 = jump2;
            jump2 = result;
        }
        return result;
    }

    /**
     * 变态跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return 0;
        }
        if (target <= 1) {
            return target;
        }
        return 2 * JumpFloorII(target - 1);
    }

    /**
     * 矩阵覆盖
     *
     * @param target
     * @return
     */
    public int RectCover(int target) {
        if (target <= 0) {
            return target;
        }
        if (target <= 2) {
            return target;
        }
        return RectCover(target - 1) + RectCover(target - 2);
    }


    /**
     * 二进制中1的个数
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int result = 0;
        while (n != 0) {
            n = n & (n - 1);
            result++;
        }
        return result;
    }

    /**
     * 数值的整次方
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent < 0) {
            base = 1 / base;
            exponent = -exponent;
        }
        return exponent % 2 == 0 ? Power(base * base, exponent / 2) :
                base * Power(base * base, exponent / 2);
    }

    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        for (int i = array.length - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                if (array[j] % 2 == 0 && array[j + 1] % 2 != 0) {
                    swap(array, j, j + 1);
                }
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 链表中倒数第k个结点
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k <= 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k - 1; i++) {
            fast = fast.next;
            if (fast == null) {
                return null;
            }
        }
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }


    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }


    public ListNode ReverseListV2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = ReverseListV2(head.next);

        head.next.next = head;

        head.next = null;

        return node;
    }

    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) {
            return null;
        }
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val <= list2.val) {
            list1.next = Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = Merge(list1, list2.next);
            return list2;
        }
    }


    /**
     * todo
     * 判断是不是子树
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return true;
        }
        if (root2 == null) {
            return false;
        }
        return false;
    }


    /**
     * 二叉树镜像
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            return;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;

        Mirror(root.left);

        Mirror(root.right);
    }


    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return result;
        }
        int left = 0;
        int right = matrix[0].length - 1;

        int top = 0;
        int bottom = matrix.length - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            top++;
            right--;
            left++;
            bottom--;
        }
        return result;
    }


}
