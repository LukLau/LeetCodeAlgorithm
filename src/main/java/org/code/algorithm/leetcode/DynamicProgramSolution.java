package org.code.algorithm.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author dora
 * @date 2020/10/7
 */
public class DynamicProgramSolution {


    /**
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = true;
                } else if (i == 0) {
                    dp[0][j] = dp[0][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
                } else if (j == 0) {
                    dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
                } else {
                    dp[i][j] = (dp[i - 1][j] && (s1.charAt(i - 1) == s3.charAt(i + j - 1))) ||
                            (dp[i][j - 1] && (s2.charAt(j - 1) == s3.charAt(i + j - 1)));
                }
            }
        }
        return dp[m][n];
    }


    /**
     * 数字之间的差异
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return -1;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0) + dp[i - 1][j];
            }
        }
        return dp[m][n];
    }

    // ---帕斯卡三角形问题---//


    /**
     * 118. Pascal's Triangle
     *
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows < 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();


        for (int i = 0; i < numRows; i++) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int j = 1; j <= i - 1; j++) {
                int value = result.get(i - 1).get(j) + result.get(i - 1).get(j - 1);
                tmp.add(value);
            }
            if (i > 0) {
                tmp.add(1);
            }
            result.add(new ArrayList<>(tmp));
        }
        return result;
    }

    /**
     * 119. Pascal's Triangle II
     *
     * @param rowIndex
     * @return
     */
    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();

        result.add(1);

        for (int i = 1; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                int val = result.get(j) + result.get(j - 1);
                result.set(j, val);
            }
            if (i > 0) {
                result.add(1);
            }
        }
        return result;
    }


    /**
     * 120. Triangle
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return 0;
        }
        int size = triangle.size();

        List<Integer> result = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {
            List<Integer> currentRow = triangle.get(i);

            int currentSize = currentRow.size();

            for (int j = 0; j < currentSize; j++) {

                int value = Math.min(result.get(j), result.get(j + 1)) + currentRow.get(j);

                result.set(j, value);
            }
        }
        return result.get(0);
    }


    // ---卖股票问题-- //

    /**
     * 121. Best Time to Buy and Sell Stock
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int minPrice = Integer.MAX_VALUE;

        int result = 0;

        for (int price : prices) {
            if (price > minPrice) {
                result = Math.max(result, price - minPrice);
            } else {
                minPrice = price;
            }
        }
        return result;
    }


    /**
     * 122. Best Time to Buy and Sell Stock II
     *
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;

        int minPrice = Integer.MAX_VALUE;

        for (int price : prices) {
            if (price > minPrice) {
                result += price - minPrice;
            }
            minPrice = price;
        }
        return result;
    }


    /**
     * 123. Best Time to Buy and Sell Stock III
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int[] leftSell = new int[prices.length];

        int[] rightSell = new int[prices.length + 1];

        int leftProfit = 0;

        int leftCost = prices[0];

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > leftCost) {
                leftProfit = Math.max(leftProfit, prices[i] - leftCost);
            } else {
                leftCost = prices[i];
            }
            leftSell[i] = leftProfit;
        }
        int rightProfit = 0;

        int rightCost = prices[prices.length - 1];

        for (int i = prices.length - 2; i >= 0; i--) {
            if (prices[i] < rightCost) {
                rightProfit = Math.max(rightProfit, rightCost - prices[i]);
            } else {
                rightCost = prices[i];
            }
            rightSell[i] = rightProfit;
        }
        int result = 0;

        for (int i = 1; i < prices.length; i++) {
            result = Math.max(result, leftSell[i] + rightSell[i + 1]);
        }
        return result;
    }


    /**
     * todo
     * 188. Best Time to Buy and Sell Stock IV
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfitIV(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int len = prices.length;

        int[] dp = new int[len];

        for (int i = 1; i < len; i++) {

            int cost = -prices[i];

            for (int j = 1; j < k; j++) {

            }
        }


        return -1;
    }


    /**
     * 132. Palindrome Partitioning II
     *
     * @param s
     * @return
     */
    public int minCut(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int n = s.length();
        boolean[][] palindrome = new boolean[n][n];
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            int minCut = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || palindrome[j + 1][i - 1])) {
                    palindrome[j][i] = true;
                    minCut = Math.min(minCut, j == 0 ? 0 : dp[j - 1] + 1);
                }
            }
            dp[i] = minCut;
        }
        return dp[n - 1];
    }


    /**
     * 134. Gas Station
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null) {
            return -1;
        }
        int remainGas = 0;
        int localGas = 0;
        int resultIndex = 0;
        for (int i = 0; i < gas.length; i++) {
            remainGas += gas[i] - cost[i];
            localGas += gas[i] - cost[i];
            if (localGas < 0) {
                resultIndex = i + 1;

                localGas = 0;
            }
        }
        return remainGas >= 0 ? resultIndex : -1;
    }


    /**
     * 135. Candy
     *
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int n = ratings.length;
        int[] dp = new int[n];

        Arrays.fill(dp, 1);

        int result = 0;

        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1] && dp[i] < dp[i - 1] + 1) {
                dp[i] = dp[i - 1] + 1;
            }
        }
        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && dp[i] < dp[i + 1] + 1) {
                dp[i] = dp[i + 1] + 1;
            }
        }


        for (int cost : dp) {
            result += cost;
        }
        return result;

    }


    /**
     * 174. Dungeon Game
     *
     * @param dungeon
     * @return
     */
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
        int row = dungeon.length;
        int column = dungeon[0].length;
        int[][] dp = new int[row][column];
        for (int i = row - 1; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (i == row - 1 && j == column - 1) {
                    dp[i][j] = Math.max(1, 1 - dungeon[i][j]);
                } else if (i == row - 1) {
                    dp[i][j] = Math.max(1, dp[i][j + 1] - dungeon[i][j]);
                } else if (j == column - 1) {
                    dp[i][j] = Math.max(1, dp[i + 1][j] - dungeon[i][j]);
                } else {
                    dp[i][j] = Math.max(1, Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]);
                }
            }
        }
        return dp[0][0];
    }


    public int calculateMinimumHPV2(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }
        int row = dungeon.length;
        int column = dungeon[0].length;
        int[] dp = new int[column];
        for (int i = column - 1; i >= 0; i--) {
            if (i == column - 1) {
                dp[i] = Math.max(1, 1 - dungeon[row - 1][i]);
            } else {
                dp[i] = Math.max(1, dp[i + 1] - dungeon[row - 1][i]);
            }
        }
        for (int i = row - 2; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (j == column - 1) {
                    dp[j] = Math.max(1, dp[j] - dungeon[i][j]);
                } else {
                    dp[j] = Math.max(1, Math.min(dp[j], dp[j + 1]) - dungeon[i][j]);
                }

            }
        }
        return dp[0];
    }


    // ---房屋大盗---//

    /**
     * 198. House Robber
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int robPre = 0;
        int robCurrent = 0;
        for (int num : nums) {
            int tmp = robPre;
            robPre = Math.max(robCurrent, robPre);
            robCurrent = tmp + num;
        }
        return Math.max(robPre, robCurrent);

    }


}
