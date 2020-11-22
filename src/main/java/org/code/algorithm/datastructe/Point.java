package org.code.algorithm.datastructe;

import lombok.Data;

/**
 * @author dora
 * @date 2020/8/18
 */
@Data
public class Point {
    public int x;
    public int y;

    public Point() {
    }


    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }



}
