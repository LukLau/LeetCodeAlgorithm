package org.code.algorithm.swordoffer;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.RandomListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Stack;

/**
 * @author dora
 * @date 2020/7/25
 */
public class SwordOffer {

    public static void main(String[] args) {
        SwordOffer swordOffer = new SwordOffer();
        int[] input = new int[]{4, 5, 1, 6, 2, 7, 3, 8};

        swordOffer.GetLeastNumbersSolution(input, 10);
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


    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int value : pushA) {
            stack.push(value);
            while (!stack.isEmpty() && popA[j] == stack.peek()) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }


    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        int end = sequence.length - 1;
        while (end > 0) {
            int index = 0;
            while (index < end && sequence[index] < sequence[end]) {
                index++;
            }
            while (index < end && sequence[index] > sequence[end]) {
                index++;
            }
            if (index != end) {
                return false;
            }
            end--;
        }
        return true;

//        return intervalVerify(0, sequence.length - 1, sequence);
    }


    public boolean VerifySquenceOfBSTV2(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        return intervalVerify(0, sequence.length - 1, sequence);
    }

    private boolean intervalVerify(int start, int end, int[] sequence) {
        if (start > end) {
            return true;
        }
        if (start == end) {
            return true;
        }
        int tmp1 = start;
        while (tmp1 < end && sequence[tmp1] < sequence[end]) {
            tmp1++;
        }
        int tmp2 = tmp1;
        while (tmp2 < end && sequence[tmp2] > sequence[end]) {
            tmp2++;
        }
        if (tmp2 != end) {
            return false;
        }
        return intervalVerify(start, tmp1 - 1, sequence) && intervalVerify(tmp2, end, sequence);
    }

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) {
            return new ArrayList<>();
        }
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        FindPath(result, new ArrayList<Integer>(), root, target);
        return result;

    }

    private void FindPath(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> integers, TreeNode root, int target) {
        integers.add(root.val);
        if (root.left == null && root.right == null && root.val == target) {
            result.add(new ArrayList<>(integers));
        } else {
            if (root.left != null) {
                FindPath(result, integers, root.left, target - root.val);
            }
            if (root.right != null) {
                FindPath(result, integers, root.right, target - root.val);
            }
        }
        integers.remove(integers.size() - 1);
    }


    /**
     * 复制链表
     *
     * @param pHead
     * @return
     */
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }
        RandomListNode node = pHead;

        while (node != null) {

            RandomListNode tmp = new RandomListNode(node.label);

            tmp.next = node.next;

            node.next = tmp;

            node = tmp.next;
        }

        node = pHead;

        while (node != null) {
            RandomListNode tmp = node.next;
            if (node.random != null) {
                tmp.random = node.random.next;
            }
            node = tmp.next;
        }

        node = pHead;

        RandomListNode cloneNode = node.next;

        while (node.next != null) {
            RandomListNode tmp = node.next;

            node.next = tmp.next;

            node = tmp;
        }
        return cloneNode;
    }


    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        TreeNode p = pRootOfTree;
        TreeNode root = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev != null) {
                prev.right = p;
                p.left = prev;
            } else {
                root = p;
            }
            prev = p;
            p = p.right;
        }
        return root;
    }


    public ArrayList<String> Permutation(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        char[] words = str.toCharArray();
        boolean[] used = new boolean[words.length];
        intervalPermutation(result, "", used, words);
        return result;
    }

    private void intervalPermutation(ArrayList<String> result, String word, boolean[] used, char[] words) {
        if (word.length() == words.length) {
            result.add(word);
            return;
        }
        for (int i = 0; i < words.length; i++) {
            if (i > 0 && words[i] == words[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }
            used[i] = true;
            word += words[i];
            intervalPermutation(result, word, used, words);
            word = word.substring(0, word.length() - 1);
            used[i] = false;
        }
    }


    /**
     * todo
     *
     * @param str
     * @return
     */
    public ArrayList<String> PermutationV2(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<String> result = new ArrayList<>();
        char[] words = str.toCharArray();
        intervalPermutationV2(result, 0, words);
        return result;

    }

    private void intervalPermutationV2(ArrayList<String> result, int start, char[] words) {
        if (start == words.length) {
            result.add(String.valueOf(words));
        }


    }


    /**
     * 摩尔投票法
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNumSolution(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int count = 1;
        int candidate = array[0];
        for (int i = 1; i < array.length; i++) {
            int val = array[i];
            if (val == candidate) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    candidate = val;
                    count = 1;
                }
            }
        }
        count = 0;
        for (int num : array) {

            if (num == candidate) {
                count++;
            }
        }
        if (count * 2 > array.length) {
            return candidate;
        }
        return 0;
    }


    /**
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbersSolution(int[] input, int k) {
        if (input == null || input.length == 0) {
            return new ArrayList<>();
        }
        if (k > input.length || k <= 0) {
            return new ArrayList<>();
        }
        Arrays.sort(input);
        ArrayList<Integer> result = new ArrayList<>();
        int count = 0;
        for (Integer num : input) {
            result.add(num);
            count++;
            if (count == k) {
                break;
            }
        }
        return result;
    }

    public ArrayList<Integer> GetLeastNumbersSolutionV2(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<>();
        if (input == null || input.length == 0 || k <= 0 || k > input.length) {
            return result;
        }
        k--;
        int partition = partition(input, 0, input.length - 1);
        while (partition != k) {

            if (partition > k) {
                partition = partition(input, 0, partition - 1);
            } else {
                partition = partition(input, partition + 1, input.length - 1);
            }
        }
        for (int i = 0; i <= k; i++) {
            result.add(input[i]);
        }
        return result;
    }

    private int partition(int[] input, int start, int end) {
        int pivot = input[start];
        while (start < end) {
            while (start < end && input[end] >= pivot) {
                end--;
            }
            if (start < end) {
                input[start] = input[end];
                start++;
            }
            while (start < end && input[start] <= pivot) {
                start++;
            }
            if (start < end) {
                input[end] = input[start];
                end--;
            }
        }
        input[start] = pivot;
        return start;
    }


    /**
     * todo
     *
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        return -1;
    }


    /**
     * todo
     *
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        return -1;

    }


    public String PrintMinNumber(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return "";
        }
        String[] words = new String[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            words[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(words, (o1, o2) -> {
            String tmp1 = o1 + o2;
            String tmp2 = o2 + o1;
            return tmp1.compareTo(tmp2);
        });
        StringBuilder builder = new StringBuilder();
        for (String word : words) {
            builder.append(word);
        }
        return builder.toString();
    }

    /**
     * 获取丑数
     *
     * @param index
     * @return
     */
    public int GetUglyNumberSolution(int index) {
        if (index <= 0) {
            return index;
        }
        if (index <= 7) {
            return index;
        }
        int index2 = 0;
        int index3 = 0;
        int index5 = 0;
        int[] result = new int[index];
        result[0] = 1;
        while (index < result.length) {
            int val = Math.min(Math.min(result[index2] * 2, result[index3] * 3), result[index5] * 5);

            if (val == result[index2] * 2) {
                index2++;
            }
            if (val == result[index3] * 3) {
                index3++;
            }
            index++;
        }
        return 0;
    }


    public int FirstNotRepeatingChar(String str) {
        if (str == null || str.isEmpty()) {
            return -1;
        }
        char[] words = str.toCharArray();
        int[] hash = new int[512];
        for (int i = 0; i < words.length; i++) {
            int index = words[i] - 'a';
            hash[index]++;
        }

        for (int i = 0; i < words.length; i++) {
            int index = words[i] - 'a';
            if (hash[index] == 1) {
                return i;
            }
        }
        return -1;


    }


}
