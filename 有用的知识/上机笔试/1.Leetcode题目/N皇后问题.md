## N皇后问题

```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.ans = []
        self.no_diagnoal = []
        self.no_column = []
        self.board_list = []
        def get_board(position,n):
            board = [["." for i in range(n)] for _ in range(n)]
            for i in range(len(position)):
                board[i][position[i]] = "Q"
            board = ["".join(_) for _ in board]
            return board

        def diagonal(i,j,n):
            diag = []
            x = i; y = j
            while x >= 0 and y >= 0:
                x -= 1; y -= 1
                diag.append((x,y))
            x = i; y = j
            while x < n and y < n:
                x += 1; y += 1
                diag.append((x,y))
            x = i; y = j
            while x < n and y >= 0:
                x += 1; y -= 1
                diag.append((x,y))
            x = i; y = j
            while x >= 0 and y < n:
                x -= 1; y += 1
                diag.append((x,y))
            return diag

        def dfs(i,n): ##正在摆第i个皇后，总共有n'个
            if i == n:
                self.board_list.append(get_board(self.ans,n))
                return 
            for j in range(n):
                
                if j not in self.no_column and (i,j) not in self.no_diagnoal:
                    self.ans.append(j)
                    self.no_column.append(j)
                    no_diag = diagonal(i,j,n)
                    tmp = self.no_diagnoal
                    for item in no_diag:
                        self.no_diagnoal.append(item)
                    dfs(i+1,n)
                    for _ in range(len(no_diag)):
                        self.no_diagnoal.pop()  ###易错！不能直接把list赋值成tmp，这样会出现浅拷贝问题！！
                    self.no_diagnoal
                    self.no_column.pop()
                    self.ans.pop()
        dfs(0,n)
        return self.board_list
```