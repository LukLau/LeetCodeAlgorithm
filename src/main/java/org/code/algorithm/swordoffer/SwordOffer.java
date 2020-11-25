package org.code.algorithm.swordoffer;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * @author dora
 * @date 2020/7/25
 */
public class SwordOffer {
//
//    public static void main(String[] args) {
//        SwordOffer swordOffer = new SwordOffer();
//        int[] input = new int[]{4, 5, 1, 6, 2, 7, 3, 8};
//
////        swordOffer.isNumeric(new char[]{'1', '2', '3', '.', '4', '5', 'e', '+', '6'});
//        int[] nums = new int[]{2, 3, 4, 2, 6, 2, 5, 1};
//        swordOffer.maxInWindows(nums, 3);
//    }
//
//    /**
//     * 二维数组中的查找
//     *
//     * @param target
//     * @param array
//     * @return
//     */
//    public boolean Find(int target, int[][] array) {
//        if (array == null || array.length == 0) {
//            return false;
//        }
//        int row = array.length;
//        int column = array[0].length;
//        int i = row - 1;
//        int j = 0;
//        while (i >= 0 && j < column) {
//            int value = array[i][j];
//
//            if (value == target) {
//                return true;
//            } else if (value < target) {
//                j++;
//            } else {
//                i--;
//            }
//        }
//        return false;
//    }
//
//    /**
//     * 替换空格
//     *
//     * @param str
//     * @return
//     */
//    public String replaceSpace(StringBuffer str) {
//        if (str == null) {
//            return "";
//        }
//        String word = str.toString();
//
//        int blankWord = 0;
//        for (char tmp : word.toCharArray()) {
//            if (tmp == ' ') {
//                blankWord++;
//            }
//        }
//        StringBuilder builder = new StringBuilder();
//
//        String[] words = word.split(" ");
//
//        for (String s : words) {
//            builder.append(s);
//            if (blankWord-- > 0) {
//                builder.append("%20");
//            }
//        }
//        while (blankWord-- > 0) {
//            builder.append("%20");
//        }
//        return builder.toString();
//    }
//
//
//    /**
//     * 从尾到头打印链表
//     *
//     * @param listNode
//     * @return
//     */
//    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
//        if (listNode == null) {
//            return new ArrayList<>();
//        }
//        LinkedList<Integer> linkedList = new LinkedList<>();
//        while (listNode != null) {
//            linkedList.addFirst(listNode.val);
//            listNode = listNode.next;
//        }
//        return new ArrayList<>(linkedList);
//    }
//
//
//    /**
//     * 前序 中序重建二叉树
//     *
//     * @param pre
//     * @param in
//     * @return
//     */
//    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
//        if (pre == null || in == null) {
//            return null;
//        }
//        return intervalConstruct(pre, 0, in, 0, in.length - 1);
//    }
//
//    private TreeNode intervalConstruct(int[] pre, int preStart, int[] in, int inStart, int inEnd) {
//        if (preStart >= pre.length || inStart > inEnd) {
//            return null;
//        }
//        TreeNode root = new TreeNode(pre[preStart]);
//        int index = 0;
//        for (int i = inStart; i <= inEnd; i++) {
//            if (in[i] == root.val) {
//                index = i;
//                break;
//            }
//        }
//        root.left = intervalConstruct(pre, preStart + 1, in, inStart, index - 1);
//        root.right = intervalConstruct(pre, preStart + index - inStart + 1, in, index + 1, inEnd);
//        return root;
//    }
//
//
//    /**
//     * 旋转数组的最小值
//     *
//     * @param array
//     * @return
//     */
//    public int minNumberInRotateArray(int[] array) {
//        if (array == null || array.length == 0) {
//            return 0;
//        }
//        int left = 0;
//        int right = array.length - 1;
//        while (left < right) {
//            if (array[left] < array[right]) {
//                return array[left];
//            }
//            int mid = left + (right - left) / 2;
//
//            if (array[left] <= array[mid]) {
//                left = mid + 1;
//            } else {
//                right = mid;
//            }
//        }
//        return array[left];
//    }
//
//
//    public int minNumberInRotateArrayV2(int[] array) {
//        if (array == null || array.length == 0) {
//            return 0;
//        }
//        int left = 0;
//        int right = array.length - 1;
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (array[mid] <= array[right]) {
//                right = mid;
//            } else {
//                left = mid + 1;
//            }
//        }
//        return array[left];
//    }
//
//
//    /**
//     * 斐波那契
//     *
//     * @param n
//     * @return
//     */
//    public int Fibonacci(int n) {
//        if (n == 0) {
//            return 0;
//        }
//        if (n == 1) {
//            return 1;
//        }
//        int sum = 0;
//        int sum1 = 0;
//        int sum2 = 1;
//        for (int i = 2; i <= n; i++) {
//            sum = sum1 + sum2;
//            sum1 = sum2;
//            sum2 = sum;
//        }
//        return sum;
//    }
//
//    /**
//     * 跳台阶
//     *
//     * @param target
//     * @return
//     */
//    public int JumpFloor(int target) {
//        if (target <= 0) {
//            return 0;
//        }
//        if (target <= 2) {
//            return target;
//        }
//        int result = 0;
//        int jump1 = 1;
//        int jump2 = 2;
//        for (int i = 3; i <= target; i++) {
//            result = jump1 + jump2;
//            jump1 = jump2;
//            jump2 = result;
//        }
//        return result;
//    }
//
//    /**
//     * 变态跳台阶
//     *
//     * @param target
//     * @return
//     */
//    public int JumpFloorII(int target) {
//        if (target <= 0) {
//            return 0;
//        }
//        if (target <= 1) {
//            return target;
//        }
//        return 2 * JumpFloorII(target - 1);
//    }
//
//    /**
//     * 矩阵覆盖
//     *
//     * @param target
//     * @return
//     */
//    public int RectCover(int target) {
//        if (target <= 0) {
//            return target;
//        }
//        if (target <= 2) {
//            return target;
//        }
//        return RectCover(target - 1) + RectCover(target - 2);
//    }
//
//
//    /**
//     * 二进制中1的个数
//     *
//     * @param n
//     * @return
//     */
//    public int NumberOf1(int n) {
//        int result = 0;
//        while (n != 0) {
//            n = n & (n - 1);
//            result++;
//        }
//        return result;
//    }
//
//    /**
//     * 数值的整次方
//     *
//     * @param base
//     * @param exponent
//     * @return
//     */
//    public double Power(double base, int exponent) {
//        if (exponent == 0) {
//            return 1;
//        }
//        if (exponent < 0) {
//            base = 1 / base;
//            exponent = -exponent;
//        }
//        return exponent % 2 == 0 ? Power(base * base, exponent / 2) :
//                base * Power(base * base, exponent / 2);
//    }
//
//    public void reOrderArray(int[] array) {
//        if (array == null || array.length == 0) {
//            return;
//        }
//        for (int i = array.length - 1; i >= 0; i--) {
//            for (int j = 0; j < i; j++) {
//                if (array[j] % 2 == 0 && array[j + 1] % 2 != 0) {
//                    swap(array, j, j + 1);
//                }
//            }
//        }
//    }
//
//    private void swap(int[] nums, int i, int j) {
//        int val = nums[i];
//        nums[i] = nums[j];
//        nums[j] = val;
//    }
//
//
//    /**
//     * 链表中倒数第k个结点
//     *
//     * @param head
//     * @param k
//     * @return
//     */
//    public ListNode FindKthToTail(ListNode head, int k) {
//        if (head == null || k <= 0) {
//            return null;
//        }
//        ListNode fast = head;
//        for (int i = 0; i < k - 1; i++) {
//            fast = fast.next;
//            if (fast == null) {
//                return null;
//            }
//        }
//        ListNode slow = head;
//        while (fast.next != null) {
//            fast = fast.next;
//            slow = slow.next;
//        }
//        return slow;
//    }
//
//
//    /**
//     * 反转链表
//     *
//     * @param head
//     * @return
//     */
//    public ListNode ReverseList(ListNode head) {
//        if (head == null || head.next == null) {
//            return head;
//        }
//        ListNode prev = null;
//        while (head != null) {
//            ListNode tmp = head.next;
//            head.next = prev;
//            prev = head;
//            head = tmp;
//        }
//        return prev;
//    }
//
//
//    public ListNode ReverseListV2(ListNode head) {
//        if (head == null || head.next == null) {
//            return head;
//        }
//        ListNode node = ReverseListV2(head.next);
//
//        head.next.next = head;
//
//        head.next = null;
//
//        return node;
//    }
//
//    public ListNode Merge(ListNode list1, ListNode list2) {
//        if (list1 == null && list2 == null) {
//            return null;
//        }
//        if (list1 == null) {
//            return list2;
//        }
//        if (list2 == null) {
//            return list1;
//        }
//        if (list1.val <= list2.val) {
//            list1.next = Merge(list1.next, list2);
//            return list1;
//        } else {
//            list2.next = Merge(list1, list2.next);
//            return list2;
//        }
//    }
//
//
//    /**
//     * 判断是不是子树
//     *
//     * @param root1
//     * @param root2
//     * @return
//     */
//    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
//        if (root1 == null || root2 == null) {
//            return false;
//        }
//        return isSubTree(root1, root2)
//                || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
//    }
//
//    private boolean isSubTree(TreeNode root1, TreeNode root2) {
//        if (root2 == null) {
//            return true;
//        }
//        if (root1 == null) {
//            return false;
//        }
//        if (root1.val == root2.val) {
//            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
//        }
//        return false;
//    }
//
//
//    /**
//     * 二叉树镜像
//     *
//     * @param root
//     */
//    public void Mirror(TreeNode root) {
//        if (root == null) {
//            return;
//        }
//        if (root.left == null && root.right == null) {
//            return;
//        }
//        TreeNode tmp = root.left;
//        root.left = root.right;
//        root.right = tmp;
//
//        Mirror(root.left);
//
//        Mirror(root.right);
//    }
//
//
//    public ArrayList<Integer> printMatrix(int[][] matrix) {
//        ArrayList<Integer> result = new ArrayList<>();
//        if (matrix == null || matrix.length == 0) {
//            return result;
//        }
//        int left = 0;
//        int right = matrix[0].length - 1;
//
//        int top = 0;
//        int bottom = matrix.length - 1;
//        while (left <= right && top <= bottom) {
//            for (int i = left; i <= right; i++) {
//                result.add(matrix[top][i]);
//            }
//            for (int i = top + 1; i <= bottom; i++) {
//                result.add(matrix[i][right]);
//            }
//            if (top != bottom) {
//                for (int i = right - 1; i >= left; i--) {
//                    result.add(matrix[bottom][i]);
//                }
//            }
//            if (left != right) {
//                for (int i = bottom - 1; i > top; i--) {
//                    result.add(matrix[i][left]);
//                }
//            }
//            top++;
//            right--;
//            left++;
//            bottom--;
//        }
//        return result;
//    }
//
//
//    public boolean IsPopOrder(int[] pushA, int[] popA) {
//        if (pushA == null || popA == null) {
//            return false;
//        }
//        Stack<Integer> stack = new Stack<>();
//        int j = 0;
//        for (int value : pushA) {
//            stack.push(value);
//            while (!stack.isEmpty() && popA[j] == stack.peek()) {
//                stack.pop();
//                j++;
//            }
//        }
//        return stack.isEmpty();
//    }
//
//
//    public boolean VerifySquenceOfBST(int[] sequence) {
//        if (sequence == null || sequence.length == 0) {
//            return false;
//        }
//        int end = sequence.length - 1;
//        while (end > 0) {
//            int index = 0;
//            while (index < end && sequence[index] < sequence[end]) {
//                index++;
//            }
//            while (index < end && sequence[index] > sequence[end]) {
//                index++;
//            }
//            if (index != end) {
//                return false;
//            }
//            end--;
//        }
//        return true;
//
////        return intervalVerify(0, sequence.length - 1, sequence);
//    }
//
//
//    public boolean VerifySquenceOfBSTV2(int[] sequence) {
//        if (sequence == null || sequence.length == 0) {
//            return false;
//        }
//        return intervalVerify(0, sequence.length - 1, sequence);
//    }
//
//    private boolean intervalVerify(int start, int end, int[] sequence) {
//        if (start > end) {
//            return true;
//        }
//        if (start == end) {
//            return true;
//        }
//        int tmp1 = start;
//        while (tmp1 < end && sequence[tmp1] < sequence[end]) {
//            tmp1++;
//        }
//        int tmp2 = tmp1;
//        while (tmp2 < end && sequence[tmp2] > sequence[end]) {
//            tmp2++;
//        }
//        if (tmp2 != end) {
//            return false;
//        }
//        return intervalVerify(start, tmp1 - 1, sequence) && intervalVerify(tmp2, end, sequence);
//    }
//
//    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
//        if (root == null) {
//            return new ArrayList<>();
//        }
//        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
//        FindPath(result, new ArrayList<Integer>(), root, target);
//        return result;
//
//    }
//
//    private void FindPath(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> integers, TreeNode root, int target) {
//        integers.add(root.val);
//        if (root.left == null && root.right == null && root.val == target) {
//            result.add(new ArrayList<>(integers));
//        } else {
//            if (root.left != null) {
//                FindPath(result, integers, root.left, target - root.val);
//            }
//            if (root.right != null) {
//                FindPath(result, integers, root.right, target - root.val);
//            }
//        }
//        integers.remove(integers.size() - 1);
//    }
//
//
//    /**
//     * 复制链表
//     *
//     * @param pHead
//     * @return
//     */
//    public RandomListNode Clone(RandomListNode pHead) {
//        if (pHead == null) {
//            return null;
//        }
//        RandomListNode node = pHead;
//
//        while (node != null) {
//
//            RandomListNode tmp = new RandomListNode(node.label);
//
//            tmp.next = node.next;
//
//            node.next = tmp;
//
//            node = tmp.next;
//        }
//
//        node = pHead;
//
//        while (node != null) {
//            RandomListNode tmp = node.next;
//            if (node.random != null) {
//                tmp.random = node.random.next;
//            }
//            node = tmp.next;
//        }
//
//        node = pHead;
//
//        RandomListNode cloneNode = node.next;
//
//        while (node.next != null) {
//            RandomListNode tmp = node.next;
//
//            node.next = tmp.next;
//
//            node = tmp;
//        }
//        return cloneNode;
//    }
//
//
//    public TreeNode Convert(TreeNode pRootOfTree) {
//        if (pRootOfTree == null) {
//            return null;
//        }
//        Stack<TreeNode> stack = new Stack<>();
//        TreeNode prev = null;
//        TreeNode p = pRootOfTree;
//        TreeNode root = null;
//        while (!stack.isEmpty() || p != null) {
//            while (p != null) {
//                stack.push(p);
//                p = p.left;
//            }
//            p = stack.pop();
//            if (prev != null) {
//                prev.right = p;
//                p.left = prev;
//            } else {
//                root = p;
//            }
//            prev = p;
//            p = p.right;
//        }
//        return root;
//    }
//
//
//    public ArrayList<String> Permutation(String str) {
//        if (str == null || str.isEmpty()) {
//            return new ArrayList<>();
//        }
//        ArrayList<String> result = new ArrayList<>();
//        char[] words = str.toCharArray();
//        boolean[] used = new boolean[words.length];
//        intervalPermutation(result, "", used, words);
//        return result;
//    }
//
//    private void intervalPermutation(ArrayList<String> result, String word, boolean[] used, char[] words) {
//        if (word.length() == words.length) {
//            result.add(word);
//            return;
//        }
//        for (int i = 0; i < words.length; i++) {
//            if (i > 0 && words[i] == words[i - 1] && !used[i - 1]) {
//                continue;
//            }
//            if (used[i]) {
//                continue;
//            }
//            used[i] = true;
//            word += words[i];
//            intervalPermutation(result, word, used, words);
//            word = word.substring(0, word.length() - 1);
//            used[i] = false;
//        }
//    }
//
//
//    /**
//     * todo
//     *
//     * @param str
//     * @return
//     */
//    public ArrayList<String> PermutationV2(String str) {
//        if (str == null || str.isEmpty()) {
//            return new ArrayList<>();
//        }
//        ArrayList<String> result = new ArrayList<>();
//        char[] words = str.toCharArray();
//        intervalPermutationV2(result, 0, words);
//        return result;
//
//    }
//
//    private void intervalPermutationV2(ArrayList<String> result, int start, char[] words) {
//        if (start == words.length) {
//            result.add(String.valueOf(words));
//        }
//    }
//
//
//    /**
//     * 摩尔投票法
//     *
//     * @param array
//     * @return
//     */
//    public int MoreThanHalfNumSolution(int[] array) {
//        if (array == null || array.length == 0) {
//            return 0;
//        }
//        int count = 1;
//        int candidate = array[0];
//        for (int i = 1; i < array.length; i++) {
//            int val = array[i];
//            if (val == candidate) {
//                count++;
//            } else {
//                count--;
//                if (count == 0) {
//                    candidate = val;
//                    count = 1;
//                }
//            }
//        }
//        count = 0;
//        for (int num : array) {
//
//            if (num == candidate) {
//                count++;
//            }
//        }
//        if (count * 2 > array.length) {
//            return candidate;
//        }
//        return 0;
//    }
//
//
//    /**
//     * @param input
//     * @param k
//     * @return
//     */
//    public ArrayList<Integer> GetLeastNumbersSolution(int[] input, int k) {
//        if (input == null || input.length == 0) {
//            return new ArrayList<>();
//        }
//        if (k > input.length || k <= 0) {
//            return new ArrayList<>();
//        }
//        Arrays.sort(input);
//        ArrayList<Integer> result = new ArrayList<>();
//        int count = 0;
//        for (Integer num : input) {
//            result.add(num);
//            count++;
//            if (count == k) {
//                break;
//            }
//        }
//        return result;
//    }
//
//    public ArrayList<Integer> GetLeastNumbersSolutionV2(int[] input, int k) {
//        ArrayList<Integer> result = new ArrayList<>();
//        if (input == null || input.length == 0 || k <= 0 || k > input.length) {
//            return result;
//        }
//        k--;
//        int partition = partition(input, 0, input.length - 1);
//        while (partition != k) {
//
//            if (partition > k) {
//                partition = partition(input, 0, partition - 1);
//            } else {
//                partition = partition(input, partition + 1, input.length - 1);
//            }
//        }
//        for (int i = 0; i <= k; i++) {
//            result.add(input[i]);
//        }
//        return result;
//    }
//
//    private int partition(int[] input, int start, int end) {
//        int pivot = input[start];
//        while (start < end) {
//            while (start < end && input[end] >= pivot) {
//                end--;
//            }
//            if (start < end) {
//                input[start] = input[end];
//                start++;
//            }
//            while (start < end && input[start] <= pivot) {
//                start++;
//            }
//            if (start < end) {
//                input[end] = input[start];
//                end--;
//            }
//        }
//        input[start] = pivot;
//        return start;
//    }
//
//
//    /**
//     * @param array
//     * @return
//     */
//    public int FindGreatestSumOfSubArray(int[] array) {
//        if (array == null || array.length == 0) {
//            return Integer.MIN_VALUE;
//        }
//        int result = Integer.MIN_VALUE;
//        int local = 0;
//        for (int num : array) {
//            local = local >= 0 ? local + num : num;
//            result = Math.max(result, local);
//        }
//        return result;
//
//    }
//
//
//    /**
//     * todo
//     *
//     * @param n
//     * @return
//     */
//    public int NumberOf1Between1AndN_Solution(int n) {
//        return -1;
//
//    }
//
//
//    public String PrintMinNumber(int[] numbers) {
//        if (numbers == null || numbers.length == 0) {
//            return "";
//        }
//        String[] words = new String[numbers.length];
//        for (int i = 0; i < numbers.length; i++) {
//            words[i] = String.valueOf(numbers[i]);
//        }
//        Arrays.sort(words, (o1, o2) -> {
//            String tmp1 = o1 + o2;
//            String tmp2 = o2 + o1;
//            return tmp1.compareTo(tmp2);
//        });
//        StringBuilder builder = new StringBuilder();
//        for (String word : words) {
//            builder.append(word);
//        }
//        return builder.toString();
//    }
//
//    /**
//     * 获取丑数
//     *
//     * @param index
//     * @return
//     */
//    public int GetUglyNumberSolution(int index) {
//        if (index < 7) {
//            return index;
//        }
//        int[] numbers = new int[index];
//        numbers[0] = 1;
//        int idx2 = 0;
//        int idx3 = 0;
//        int idx5 = 0;
//        for (int i = 1; i < numbers.length; i++) {
//            int val = Math.min(Math.min(numbers[idx2] * 2, numbers[idx3] * 3), numbers[idx5] * 5);
//
//            numbers[i] = val;
//
//            if (val == numbers[idx2] * 2) {
//                idx2++;
//            }
//            if (val == numbers[idx3] * 3) {
//                idx3++;
//            }
//            if (val == numbers[idx5] * 5) {
//                idx5++;
//            }
//        }
//        return numbers[index - 1];
//    }
//
//
//    public int FirstNotRepeatingChar(String str) {
//        if (str == null || str.isEmpty()) {
//            return -1;
//        }
//        HashMap<Character, Integer> map = new HashMap<>();
//        char[] words = str.toCharArray();
//        for (char word : words) {
//            Integer num = map.getOrDefault(word, 0);
//            num++;
//            map.put(word, num);
//        }
//        for (int i = 0; i < words.length; i++) {
//            char word = words[i];
//            Integer integer = map.get(word);
//            if (integer == 1) {
//                return i;
//            }
//        }
//        return -1;
//    }
//
//
//    /**
//     * todo
//     *
//     * @param array
//     * @return
//     */
//    public int InversePairs(int[] array) {
//        return -1;
//    }
//
//
//    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
//        ListNode l1 = pHead1;
//
//        ListNode l2 = pHead2;
//        while (l1 != l2) {
//            l1 = l1 == null ? pHead2 : l1.next;
//            l2 = l2 == null ? pHead1 : l2.next;
//        }
//        return l1;
//    }
//
//
//    public int GetNumberOfK(int[] array, int k) {
//        if (array == null || array.length == 0) {
//            return 0;
//        }
//        int count = 0;
//        for (int num : array) {
//            if (num == k) {
//                count++;
//            }
//
//        }
//        return count;
//
//    }
//
//
//    /**
//     * todo
//     *
//     * @param array
//     * @param k
//     * @return
//     */
//    public int GetNumberOfKV2(int[] array, int k) {
//        if (array == null || array.length == 0) {
//            return 0;
//        }
//        return -1;
//    }
//
//
//    public int TreeDepth(TreeNode root) {
//        if (root == null) {
//            return 0;
//        }
//        return 1 + Math.max(TreeDepth(root.left), TreeDepth(root.right));
//    }
//
//
//    public boolean IsBalancedSolution(TreeNode root) {
//
//        if (root == null) {
//            return true;
//        }
//        int leftDepth = TreeDepth(root.left);
//        int rightDepth = TreeDepth(root.right);
//        if (Math.abs(leftDepth - rightDepth) <= 1) {
//            return IsBalancedSolution(root.left) && IsBalancedSolution(root.right);
//        }
//        return false;
//
//    }
//
//
//    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
//        if (array == null || array.length == 0) {
//            return;
//        }
//        int result = 0;
//        for (int number : array) {
//            result ^= number;
//        }
//        int index = 0;
//        while (index < 32) {
//            if ((result & (1 << index)) != 0) {
//                break;
//            }
//            index++;
//        }
//        for (int number : array) {
//            if ((number & (1 << index)) == 0) {
//                num1[0] ^= number;
//            } else {
//                num2[0] ^= number;
//            }
//        }
//    }
//
//
//    /**
//     * todo
//     *
//     * @param sum
//     * @return
//     */
//    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
//        return null;
//    }
//
//
//    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
//        if (array == null || array.length == 0) {
//            return new ArrayList<>();
//        }
//        ArrayList<Integer> result = new ArrayList<>();
//        int left = 0;
//        int right = array.length - 1;
//        while (left < right) {
//            if (array[left] + array[right] == sum) {
//                result.add(array[left]);
//                result.add(array[right]);
//                break;
//            }
//            while (left < right && array[left] + array[right] < sum) {
//                left++;
//            }
//            while (left < right && array[left] + array[right] > sum) {
//                right--;
//            }
//        }
//        return result;
//    }
//
//
//    public String LeftRotateString(String str, int n) {
//        if (str == null || str.isEmpty()) {
//            return "";
//        }
//        if (n == 0) {
//            return str;
//        }
//        int len = str.length();
//        str += str;
//        return str.substring(n, n + len);
//
//    }
//
//
//    public String LeftRotateStringV2(String str, int n) {
//        if (str == null || str.isEmpty() || n <= 0) {
//            return str;
//        }
//        char[] words = str.toCharArray();
//        swapChars(words, 0, n - 1);
//        swapChars(words, n, words.length - 1);
//        swapChars(words, 0, words.length - 1);
//        return String.valueOf(words);
//    }
//
//    private void swapChars(char[] words, int start, int end) {
//        if (start > end) {
//            return;
//        }
//        for (int i = start; i <= (start + end) / 2; i++) {
//            char tmp = words[i];
//            words[i] = words[start + end - i];
//            words[start + end - i] = tmp;
//        }
//    }
//
//
//    /**
//     * @param str
//     * @return
//     */
//    public String ReverseSentence(String str) {
//        if (str == null || str.isEmpty()) {
//            return "";
//        }
//
//        String[] words = str.split(" ");
//        StringBuilder builder = new StringBuilder();
//
//        for (int i = words.length - 1; i >= 0; i--) {
//            builder.append(words[i]);
//            if (i > 0) {
//                builder.append(" ");
//            }
//        }
//        return builder.length() != 0 ? builder.toString() : str;
//    }
//
//
//    public boolean isContinuous(int[] numbers) {
//        if (numbers == null || numbers.length == 0) {
//            return false;
//        }
//        int min = 14;
//        int max = -1;
//
//        int zeroNumber = 0;
//
//        for (int number : numbers) {
//            if (number == 0) {
//                zeroNumber++;
//                continue;
//            }
//            if (min == number || max == number) {
//                return false;
//            }
//            if (number < min) {
//                min = number;
//            }
//            if (number > max) {
//                max = number;
//            }
//
//
//        }
//        if (zeroNumber >= 5) {
//            return true;
//        }
//
//        return max - min <= 4;
//    }
//
//
//    public boolean isContinuousV2(int[] numbers) {
//        if (numbers == null || numbers.length == 0) {
//            return false;
//        }
//        Arrays.sort(numbers);
//
//        if (numbers[numbers.length - 1] == 0) {
//            return true;
//        }
//        int min = 14;
//        int max = -1;
//
//        for (int i = 0; i < numbers.length; i++) {
//            int val = numbers[i];
//            if (val == 0) {
//                continue;
//            }
//            if (i > 0 && numbers[i] == numbers[i - 1]) {
//                return false;
//            }
//            if (val < min) {
//                min = val;
//            }
//            if (val > max) {
//                max = val;
//            }
//        }
//        return max - min <= 4;
//
//    }
//
//    /**
//     * todo
//     * 约瑟夫环
//     *
//     * @param n
//     * @param m
//     * @return
//     */
//    public int LastRemainingSolution(int n, int m) {
//        if (n <= 0 || m <= 0) {
//            return 0;
//        }
//        List<Integer> result = new ArrayList<>();
//        for (int i = 0; i < n; i++) {
//            result.add(i);
//        }
//        int position = 0;
//        while (result.size() > 1) {
//            position = (position + m - 1) % result.size();
//
//            result.remove(position);
//        }
//        return result.size() == 1 ? result.get(0) : -1;
//    }
//
//
//    public int Sum_Solution(int n) {
//        int result = n;
//
//        boolean flag = n > 0 && ((result += Sum_Solution(n - 1)) > 0);
//
//        return result;
//    }
//
//
//    /**
//     * todo
//     *
//     * @param num1
//     * @param num2
//     * @return
//     */
//    public int Add(int num1, int num2) {
//        return -1;
//    }
//
//
//    public int StrToInt(String str) {
//        if (str == null) {
//            return 0;
//        }
//        str = str.trim();
//
//        if (str.isEmpty()) {
//            return 0;
//        }
//        int sign = 1;
//        int index = 0;
//        if (str.charAt(index) == '-' || str.charAt(index) == '+') {
//            sign = str.charAt(index) == '-' ? -1 : 1;
//            index++;
//        }
//        long result = 0;
//        while (index < str.length() && Character.isDigit(str.charAt(index))) {
//            int val = Character.getNumericValue(str.charAt(index));
//
//            result = result * 10 + val;
//
//            if (result > Integer.MAX_VALUE) {
//                return 0;
//            }
//            index++;
//        }
//        if (index != str.length()) {
//            return 0;
//        }
//        return (int) (result * sign);
//    }
//
//
//    /**
//     * todo
//     *
//     * @param numbers
//     * @param length
//     * @param duplication
//     * @return
//     */
//    public boolean duplicate(int numbers[], int length, int[] duplication) {
//        return false;
//    }
//
//
//    public int[] multiply(int[] A) {
//        if (A == null || A.length == 0) {
//            return new int[]{};
//        }
//        int[] result = new int[A.length];
//        int base = 1;
//        for (int i = 0; i < A.length; i++) {
//            result[i] = base;
//            base *= A[i];
//        }
//        base = 1;
//        for (int i = A.length - 1; i >= 0; i--) {
//            result[i] *= base;
//            base *= A[i];
//        }
//        return result;
//    }
//
//    /**
//     * 魔法匹配问题
//     *
//     * @param str
//     * @param pattern
//     * @return
//     */
//    public boolean match(char[] str, char[] pattern) {
//        if (str == null) {
//            return true;
//        }
//        if (pattern == null) {
//            return false;
//        }
//        boolean[][] dp = new boolean[str.length + 1][pattern.length + 1];
//
//        dp[0][0] = true;
//
//        for (int j = 1; j <= pattern.length; j++) {
//            dp[0][j] = pattern[j - 1] == '*' && dp[0][j - 2];
//        }
//        for (int i = 1; i <= str.length; i++) {
//            for (int j = 1; j <= pattern.length; j++) {
//                if (str[i - 1] == pattern[j - 1] || pattern[j - 1] == '.') {
//                    dp[i][j] = dp[i - 1][j - 1];
//                } else if (pattern[j - 1] == '*') {
//                    if (pattern[j - 2] != '.' && str[i - 1] != pattern[j - 2]) {
//                        dp[i][j] = dp[i][j - 2];
//                    } else {
//                        dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 1] || dp[i - 1][j];
//                    }
//                }
//            }
//        }
//        return dp[str.length][pattern.length];
//    }
//
//
//    public boolean isNumeric(char[] str) {
//        if (str == null || str.length == 0) {
//            return false;
//        }
//        boolean seenNumber = false;
//        boolean seenE = false;
//        boolean seenDigit = false;
//        boolean seenNumberAfter = true;
//        for (int i = 0; i < str.length; i++) {
//            char word = str[i];
//            if (Character.isDigit(word)) {
//                seenNumber = true;
//
//                seenNumberAfter = true;
//            } else if (word == 'e' || word == 'E') {
//                if (i == 0 || seenE) {
//                    return false;
//                }
//                if (!Character.isDigit(str[i - 1])) {
//                    return false;
//                }
//                seenE = true;
//                seenNumberAfter = false;
//            } else if (word == '-' || word == '+') {
//                if (i > 0 && (str[i - 1] != 'e' && str[i - 1] != 'E')) {
//                    return false;
//                }
//            } else if (word == '.') {
//                if (seenDigit || i == 0 | seenE) {
//                    return false;
//                }
//                seenDigit = true;
//            } else {
//                return false;
//            }
//        }
//        return seenNumberAfter && seenNumber;
//    }
//
//
//    public ListNode EntryNodeOfLoop(ListNode pHead) {
//        if (pHead == null || pHead.next == null) {
//            return null;
//        }
//        ListNode fast = pHead;
//        ListNode slow = pHead;
//        while (fast.next != null && fast.next.next != null) {
//            fast = fast.next.next;
//            slow = slow.next;
//            if (slow == fast) {
//                fast = pHead;
//                while (fast != slow) {
//                    fast = fast.next;
//                    slow = slow.next;
//                }
//                return fast;
//            }
//        }
//        return null;
//    }
//
//
//    public ListNode deleteDuplication(ListNode pHead) {
//        if (pHead == null || pHead.next == null) {
//            return pHead;
//        }
//        if (pHead.val == pHead.next.val) {
//            ListNode node = pHead.next.next;
//            while (node != null && node.val == pHead.val) {
//                node = node.next;
//            }
//            return deleteDuplication(node);
//        } else {
//            pHead.next = deleteDuplication(pHead.next);
//            return pHead;
//        }
//    }
//
//
//    public TreeLinkNode GetNext(TreeLinkNode pNode) {
//        if (pNode == null) {
//            return null;
//        }
//        TreeLinkNode right = pNode.right;
//        if (right != null) {
//            while (right.left != null) {
//                right = right.left;
//            }
//            return right;
//        }
////        if (pNode.next.left == pNode) {
////            return pNode.next;
////        }
////        TreeLinkNode parentNode = pNode.next;
////        while (parentNode.next.left == parentNode) {
////            parentNode = parentNode.next;
////        }
//
//        while (pNode.next != null) {
//            if (pNode.next.left == pNode) {
//                return pNode.next;
//            }
//            pNode = pNode.next;
//        }
//        return null;
//    }
//
//
//    boolean isSymmetrical(TreeNode pRoot) {
//        if (pRoot == null) {
//            return true;
//        }
//        return isSymmetrical(pRoot.left, pRoot.right);
//
//    }
//
//    private boolean isSymmetrical(TreeNode left, TreeNode right) {
//        if (left == null && right == null) {
//            return true;
//        }
//        if (left == null || right == null) {
//            return false;
//        }
//        if (left.val != right.val) {
//            return false;
//        }
//        return isSymmetrical(left.left, right.right) && isSymmetrical(left.right, right.left);
//    }
//
//
//    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
//        if (pRoot == null) {
//            return new ArrayList<>();
//        }
//        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
//        LinkedList<TreeNode> linkedList = new LinkedList<>();
//        boolean leftToRight = true;
//        linkedList.offer(pRoot);
//        while (!linkedList.isEmpty()) {
//            int size = linkedList.size();
//            int[] nums = new int[size];
//            for (int i = 0; i < size; i++) {
//                TreeNode poll = linkedList.poll();
//
//                int index = leftToRight ? i : size - 1 - i;
//
//                nums[index] = poll.val;
//
//                if (poll.left != null) {
//                    linkedList.offer(poll.left);
//                }
//
//                if (poll.right != null) {
//                    linkedList.offer(poll.right);
//                }
//            }
//            leftToRight = !leftToRight;
//            ArrayList<Integer> tmp = new ArrayList<>();
//
//            for (int num : nums) {
//                tmp.add(num);
//            }
//            result.add(tmp);
//        }
//        return result;
//
//    }
//
//
//    public TreeNode KthNode(TreeNode pRoot, int k) {
//        if (pRoot == null || k <= 0) {
//            return null;
//        }
//        Stack<TreeNode> stack = new Stack<>();
//        TreeNode p = pRoot;
//        int iteratorIndex = -1;
//        k--;
//        while (!stack.isEmpty() || p != null) {
//            while (p != null) {
//                stack.push(p);
//                p = p.left;
//            }
//            p = stack.pop();
//
//            iteratorIndex++;
//
//            if (iteratorIndex == k) {
//                return p;
//            }
//            p = p.right;
//        }
//        return null;
//    }
//
//
//    public ArrayList<Integer> maxInWindows(int[] num, int size) {
//        if (num == null || num.length == 0 || size <= 0 || size > num.length) {
//            return new ArrayList<>();
//        }
//        ArrayList<Integer> result = new ArrayList<>();
//
//        LinkedList<Integer> list = new LinkedList<>();
//
//        for (int i = 0; i < num.length; i++) {
//            int k = i - size + 1;
//
//            if (!list.isEmpty() && list.peekFirst() < k) {
//                list.pollFirst();
//            }
//            while (!list.isEmpty() && num[list.peekLast()] <= num[i]) {
//                list.pollLast();
//            }
//            list.add(i);
//
//            if (k >= 0) {
//                result.add(num[list.peekFirst()]);
//            }
//        }
//        return result;
//    }
//
//
//    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
//        if (matrix == null || matrix.length == 0) {
//            return false;
//        }
//        boolean[][] used = new boolean[rows][cols];
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                int index = i * cols + j;
//                if (matrix[index] == str[0] && intervalHasPath(used, i, j, rows, cols, matrix, 0, str)) {
//                    return true;
//                }
//            }
//        }
//        return false;
//    }
//
//    private boolean intervalHasPath(boolean[][] used, int i, int j, int rows, int cols, char[] matrix, int start, char[] str) {
//        if (start == str.length) {
//            return true;
//        }
//        if (i < 0 || i >= rows || j < 0 || j >= cols || used[i][j]) {
//            return false;
//        }
//        int index = i * cols + j;
//
//        if (matrix[index] != str[start]) {
//            return false;
//        }
//        used[i][j] = true;
//        if (intervalHasPath(used, i - 1, j, rows, cols, matrix, start + 1, str) ||
//                intervalHasPath(used, i + 1, j, rows, cols, matrix, start + 1, str) ||
//                intervalHasPath(used, i, j - 1, rows, cols, matrix, start + 1, str) ||
//                intervalHasPath(used, i, j + 1, rows, cols, matrix, start + 1, str)) {
//            return true;
//        }
//        used[i][j] = false;
//        return false;
//    }
//
//
//    /**
//     * todo
//     *
//     * @param threshold
//     * @param rows
//     * @param cols
//     * @return
//     */
//    public int movingCount(int threshold, int rows, int cols) {
//        if (rows <= 0 || cols <= 0 || threshold < 0) {
//            return 0;
//        }
//        boolean[][] used = new boolean[rows][cols];
//        return intervalMovingCount(used, 0, 0, rows, cols, threshold);
//    }
//
//    private int intervalMovingCount(boolean[][] used, int i, int j, int rows, int cols, int threshold) {
//        if (i < 0 || i >= rows || j < 0 || j >= cols) {
//            return 0;
//        }
//        if (used[i][j]) {
//            return 0;
//        }
//        used[i][j] = true;
//        boolean notExceed = (this.getCount(i) + this.getCount(j)) <= threshold;
//
//        if (!notExceed) {
//            return 0;
//        }
//        int count = 1;
//
//        count += intervalMovingCount(used, i - 1, j, rows, cols, threshold);
//        count += intervalMovingCount(used, i + 1, j, rows, cols, threshold);
//        count += intervalMovingCount(used, i, j - 1, rows, cols, threshold);
//        count += intervalMovingCount(used, i, j + 1, rows, cols, threshold);
//
//        return count;
//    }
//
//    private int getCount(int num) {
//        int result = 0;
//        while (num != 0) {
//            result += num % 10;
//
//            num /= 10;
//        }
//        return result;
//    }
//
//
//    public int cutRope(int target) {
//        if (target <= 0) {
//            return 0;
//        }
//        if (target <= 2) {
//            return 1;
//        }
//        if (target == 3) {
//            return 2;
//        }
//        if (target == 4) {
//            return 4;
//        }
//        int[] dp = new int[target + 1];
//
//        dp[0] = dp[1];
//
//        dp[2] = 2;
//
//        dp[3] = 3;
//
//        dp[4] = 4;
//
//
//        for (int i = 5; i <= target; i++) {
//            int result = 0;
//            for (int j = 1; j <= i / 2; j++) {
//                result = Math.max(result, dp[j] * dp[i - j]);
//            }
//            dp[i] = result;
//        }
//        return dp[target];
//    }
//
//


    public static void main(String[] args) {
        SwordOffer swordOffer = new SwordOffer();
        StringBuffer buffer = new StringBuffer("hello world ");
        swordOffer.replaceSpace(buffer);
    }

    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int val = array[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    public String replaceSpace(StringBuffer str) {
        String s = str.toString();
        String insertWord = "%20";
        if (s.isEmpty()) {
            return insertWord;
        }
        int endIndex = 0;
        while (endIndex < str.length()) {
            int index = str.indexOf(" ");
            if (index == -1) {
                return str.toString();
            }
            str.replace(index, index + 1, insertWord);
            endIndex = index + 1;
        }
        return str.toString();
    }


    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {

        ArrayList<Integer> result = new ArrayList<>();
        ListNode prev = reverse(listNode);
        while (prev != null) {
            result.add(prev.val);
            prev = prev.next;
        }
        return result;
    }


    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || pre.length == 0 || in == null || in.length == 0) {
            return null;
        }
        return reConstructBinaryTree(0, pre, 0, in.length - 1, in);
    }

    private TreeNode reConstructBinaryTree(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
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
        root.left = reConstructBinaryTree(preStart + 1, pre, inStart, index - 1, in);
        root.right = reConstructBinaryTree(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
        return root;
    }


    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
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

    public int JumpFloorII(int target) {
        if (target <= 2) {
            return target;
        }
        return 2 * JumpFloorII(target - 1);
    }


    public int NumberOf1(int n) {
        int result = 0;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }


    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent < 0) {
            base = 1 / base;
            exponent = -exponent;
        }
        return exponent % 2 == 0 ? Power(base * base, exponent / 2) : base * Power(base * base, exponent / 2);
    }

    public double PowerV2(double base, int exponent) {
        long abs = Math.abs((long) exponent);
        double result = 1;
        while (abs != 0) {
            if (abs % 2 != 0) {
                result *= base;
            }
            base *= base;
            abs >>= 1;
        }
        return exponent < 0 ? 1 / result : result;
    }


    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        Arrays.sort(array);
        int len = array.length;
        int[] result = new int[len];
        int index = 0;
        for (int j : array) {
            if (j % 2 == 1) {
                result[index++] = j;
            }
        }
        for (int j : array) {
            if (j % 2 == 0) {
                result[index++] = j;
            }
        }
        System.arraycopy(result, 0, array, 0, result.length);
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k < 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k - 1; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;

    }

    private ListNode reverse(ListNode listNode) {
        ListNode prev = null;
        while (listNode != null) {
            ListNode tmp = listNode.next;
            listNode.next = prev;
            prev = listNode;
            listNode = tmp;
        }
        return prev;
    }


    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode listNode = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return listNode;
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


    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return false;
        }
        return intervalIsSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }

    private boolean intervalIsSubTree(TreeNode root1, TreeNode root2) {
        if (root2 == null) {
            return true;
        }
        if (root1 == null) {
            return false;
        }
        if (root1.val == root2.val) {
            return intervalIsSubTree(root1.left, root2.left) && intervalIsSubTree(root1.right, root2.right);
        }
        return false;
    }


    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;
        Mirror(root.left);
        Mirror(root.right);
    }


    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null) {
            return false;
        }
        return false;

    }
}
