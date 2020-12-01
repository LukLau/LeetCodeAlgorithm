package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeLinkNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * 数学解决方案
 *
 * @author luk
 * @date 2020/10/15
 */
public class MathSolution {

    public static void main(String[] args) {
        MathSolution solution = new MathSolution();
        int[] nums = new int[]{3, 2, 3};

        int n = 19;

        solution.isHappy(8);
    }


    // --- 单个数字问题 ---//

    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {

            result ^= num;
        }
        return result;
    }

    /**
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberV2(int[] nums) {
        return -1;
    }


    /**
     * 260. Single Number III
     *
     * @param nums
     * @return
     */
    public int[] singleNumberIII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
//        int baseIndex = 0;
//        for (int i = 0; i < 32; i++) {
//            if ((result & (1 << i)) != 0) {
//                baseIndex = i;
//                break;
//            }
//        }
        result &= -result;
        int[] answer = new int[2];
        for (int num : nums) {
            int tmp = num & result;
            if (tmp != 0) {
                answer[0] ^= num;
            } else {
                answer[1] ^= num;
            }
        }
        return answer;
    }


    /**
     * 149. Max Points on a Line
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }

        int result = 0;
        for (int i = 0; i < points.length; i++) {
            HashMap<Integer, Map<Integer, Integer>> map = new HashMap<>();
            int repeatPoint = 0;
            int distinctPoint = 0;
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];

                if (x == 0 && y == 0) {
                    repeatPoint++;
                    continue;
                }
                int gcd = gcd(x, y);
                x /= gcd;
                y /= gcd;
                if (map.containsKey(x)) {
                    Map<Integer, Integer> vertical = map.get(x);

                    if (vertical.containsKey(y)) {
                        Integer point = vertical.get(y);
                        vertical.put(y, point + 1);
                    } else {
                        vertical.put(y, 1);
                    }
                } else {
                    Map<Integer, Integer> vertical = new HashMap<>();
                    vertical.put(y, 1);
                    map.put(x, vertical);
                }
                distinctPoint = Math.max(distinctPoint, map.get(x).get(y));
            }
            result = Math.max(result, distinctPoint + repeatPoint + 1);
        }
        return result;
    }

    private int gcd(int x, int y) {
        if (y == 0) {
            return x;
        }
        return gcd(y, x % y);
    }

    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int maxValue = nums[0];
        int minValue = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int currentValue = nums[i];
            int tmpMaxValue = Math.max(Math.max(maxValue * currentValue, minValue * currentValue), currentValue);
            int tmpMinValue = Math.min(Math.min(maxValue * currentValue, minValue * currentValue), currentValue);

            result = Math.max(result, tmpMaxValue);
            maxValue = tmpMaxValue;
            minValue = tmpMinValue;
        }
        return result;
    }


    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        // write your code here
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        int diff = Math.abs(m - n);
        if (diff > 1) {
            return false;
        }
        if (m < n) {
            return isOneEditDistance(t, s);
        }
        if (diff == 1) {
            for (int i = 0; i < n; i++) {
                if (s.charAt(i) != t.charAt(i)) {
                    return s.substring(i + 1).equals(t.substring(i));
                }
            }
        }
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                count++;
            }
            if (count > 1) {
                return false;
            }
        }
        return true;
    }


    /**
     * 163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        // write your code here
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        for (int num : nums) {
            if (num > lower) {
                String range = constructRange(lower, num - 1);

                result.add(range);
            }
            if (num == upper) {
                return result;
            }
            lower = num + 1;
        }
        if (lower <= upper) {
            String word = constructRange(lower, upper);
            result.add(word);
        }
        return result;
    }

    private String constructRange(int start, int end) {
        return start == end ? start + "" : start + "->" + end;
    }


    /**
     * moore vote
     * 摩尔投票法
     * 169. Majority Element
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 0;
        for (int num : nums) {
            if (num == candidate) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                candidate = num;
                count = 1;
            }
        }
        return candidate;
    }


    /**
     * todo
     * 229. Majority Element II
     *
     * @param nums
     * @return
     */
    public List<Integer> majorityElementII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        int countA = 0;
        int countB = 0;
        int candidateA = nums[0];
        int candidateB = nums[0];
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
                continue;
            }
            if (num == candidateB) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = num;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = num;
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        List<Integer> result = new ArrayList<>();
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
            } else if (num == candidateB) {
                countB++;
            }
        }
        if (countA * 3 > nums.length) {
            result.add(candidateA);
        }
        if (countB * 3 > nums.length) {
            result.add(candidateB);
        }
        return result;
    }


    /**
     * 根据5的个数有关
     * 172. Factorial Trailing Zeroes
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;
        while ((n / 5) != 0) {
            count += (n / 5);
            n /= 5;
        }
        return count;
    }


    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }
        String[] words = new String[nums.length];
        for (int i = 0; i < words.length; i++) {
            words[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(words, (o1, o2) -> {
            String word1 = o1 + o2;
            String word2 = o2 + o1;
            return word2.compareTo(word1);
        });
        if ("0".equals(words[0])) {
            return "0";
        }
        StringBuilder builder = new StringBuilder();
        for (String word : words) {
            builder.append(word);
        }
        return builder.toString();
    }


    /**
     * todo
     * 187. Repeated DNA Sequences
     *
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequences(String s) {
        return null;
    }


    /**
     * 190. Reverse Bits
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        return -1;
    }

    /**
     * 一个数  二进制中1的个数
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;
    }


    /**
     * todo
     * 201. Bitwise AND of Numbers Range
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        return -1;
    }


    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        Set<Integer> repeat = new HashSet<>();
        while (n != 1) {
            int tmp = n;
            int value = 0;

            while (tmp != 0) {
                int remain = tmp % 10;

                value += remain * remain;

                tmp /= 10;
            }
            if (value == 1) {
                return true;
            }

            if (repeat.contains(value)) {
                return false;
            }
            repeat.add(value);
            n = value;
        }
        return true;
    }


    /**
     * 231. Power of Two
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        if (n == 0) {
            return false;
        }
        while (n != 0) {
            int tmp = n & (n - 1);
            if (tmp != 0) {
                return false;
            }
            n >>= 1;
        }
        return true;
    }

    /**
     * todo
     * 233. Number of Digit One
     *
     * @param n
     * @return
     */
    public int countDigitOne(int n) {
        return -1;
    }


    public int addDigits(int num) {
        return 1 + (num - 1) % 9;

    }


    public int missingNumber(int[] nums) {
        int len = nums.length;
        int result = (len) * (len + 1) / 2;
        for (int num : nums) {
            result -= num;
        }
        return result;
    }


    // --- 丑数系列 ---//

    /**
     * todo
     * 263. Ugly Number
     *
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num < 7) {
            return true;
        }
        return false;
    }


    /**
     * todo 使用曼哈顿距离
     * 296 Best Meeting Point
     *
     * @param grid: a 2D grid
     * @return: the minimize travel distance
     */
    public int minTotalDistance(int[][] grid) {
        // Write your code here
        // Write your code here
        if (grid == null || grid.length == 0) {
            return -1;
        }
        int row = grid.length;
        int column = grid[0].length;
        return -1;
    }


    /**
     * 稀疏矩阵算法
     * todo Sparse Matrix Multiplication
     *
     * @param A: a sparse matrix
     * @param B: a sparse matrix
     * @return: the result of A * B
     */
    public int[][] multiply(int[][] A, int[][] B) {
        // write your code here
        int rowA = A.length;
        int columnA = A[0].length;
        int columnB = B[0].length;
        int[][] result = new int[rowA][columnB];
        for (int i = 0; i < rowA; i++) {
            for (int k = 0; k < columnA; k++) {
                if (A[i][k] == 0) {
                    continue;
                }
                for (int j = 0; j < columnB; j++) {
                    result[i][j] += A[i][k] * B[k][j];
                }

            }
        }
        return result;
    }


    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        ListNode next = pHead.next;
        if (pHead.val == next.val) {
            while (next != null && next.val == pHead.val) {
                next = next.next;
            }
            return deleteDuplication(next);
        }
        pHead.next = deleteDuplication(pHead.next);
        return pHead;
    }


    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }
        TreeLinkNode nextNode = pNode.right;
        if (nextNode != null) {
            while (nextNode.left != null) {
                nextNode = nextNode.left;
            }
            return nextNode;
        }
        while (pNode.next != null) {
            if (pNode.next.left == pNode) {
                return pNode.next;
            }
            pNode = pNode.next;
        }
        return null;
    }

    public boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null) {
            return true;
        }
        return intervalSymmetrical(pRoot.left, pRoot.right);
    }

    private boolean intervalSymmetrical(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return intervalSymmetrical(left.left, right.right) && intervalSymmetrical(left.right, right.left);
    }


}
