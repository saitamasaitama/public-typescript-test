class ReversiBoard {
  size: number = 8;
  board: number[][];
  currentPlayer: number = 1; // 1=黒, -1=白

  directions = [
    [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [1, -1], [-1, 1], [-1, -1]
  ];

  constructor(size: number = 8) {
    this.size = size;
    this.board = Array.from({ length: size }, () => Array(size).fill(0));
    this.init();
  }

  init() {
    const mid = this.size / 2;
    this.board[mid - 1][mid - 1] = -1;
    this.board[mid][mid] = -1;
    this.board[mid - 1][mid] = 1;
    this.board[mid][mid - 1] = 1;
  }

  clone(): ReversiBoard {
    const copy = new ReversiBoard(this.size);
    // 盤面をディープコピー
    copy.board = this.board.map(row => [...row]);
    copy.currentPlayer = this.currentPlayer;
    return copy;
  }

  inBounds(x: number, y: number): boolean {
    return x >= 0 && x < this.size && y >= 0 && y < this.size;
  }

  isValidMove(x: number, y: number, player: number = this.currentPlayer): boolean {
    if (this.board[y][x] !== 0) return false;

    for (const [dx, dy] of this.directions) {
      let nx = x + dx;
      let ny = y + dy;
      let foundOpponent = false;

      while (this.inBounds(nx, ny) && this.board[ny][nx] === -player) {
        foundOpponent = true;
        nx += dx; ny += dy;
      }

      if (foundOpponent &&
          this.inBounds(nx, ny) &&
          this.board[ny][nx] === player) {
        return true;
      }
    }
    return false;
  }

  getValidMoves(player: number = this.currentPlayer): [number, number][] {
    const moves: [number, number][] = [];
    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        if (this.isValidMove(x, y, player)) {
          moves.push([x, y]);
        }
      }
    }
    return moves;
  }

  applyMove(x: number, y: number, player: number = this.currentPlayer): boolean {
    if (!this.isValidMove(x, y, player)) return false;

    this.board[y][x] = player;

    for (const [dx, dy] of this.directions) {
      const flips: [number, number][] = [];
      let nx = x + dx;
      let ny = y + dy;

      while (this.inBounds(nx, ny) && this.board[ny][nx] === -player) {
        flips.push([nx, ny]);
        nx += dx; ny += dy;
      }

      if (this.inBounds(nx, ny) && this.board[ny][nx] === player) {
        for (const [fx, fy] of flips) {
          this.board[fy][fx] = player;
        }
      }
    }

    this.switchPlayer();
    return true;
  }

  switchPlayer() {
    this.currentPlayer = -this.currentPlayer;
  }

  isGameOver(): boolean {
    return this.getValidMoves(1).length === 0 &&
           this.getValidMoves(-1).length === 0;
  }

  countStones() {
    let black = 0, white = 0;
    for (const row of this.board) {
      for (const cell of row) {
        if (cell === 1) black++;
        if (cell === -1) white++;
      }
    }
    return { black, white };
  }
}

type Strategy = "random" | "greedy" | "minimax";

class ReversiAI {
  strategy: Strategy;
  maxDepth: number;

  constructor(strategy: Strategy = "greedy", maxDepth: number = 3) {
    this.strategy = strategy;
    this.maxDepth = maxDepth;
  }

  getBestMove(board: ReversiBoard, player: number): [number, number] | null {
    const moves = board.getValidMoves(player);
    if (moves.length === 0) return null;

    switch (this.strategy) {
      case "random":
        return this.chooseRandom(moves);
      case "greedy":
        return this.chooseGreedy(board, player, moves);
      case "minimax":
        return this.chooseMinimax(board, player, moves);
      default:
        return this.chooseRandom(moves);
    }
  }

  // ---------- ランダム戦略 ----------
  private chooseRandom(moves: [number, number][]): [number, number] {
    const idx = Math.floor(Math.random() * moves.length);
    return moves[idx];
  }

  // ---------- 貪欲戦略：その一手で一番多くの石が増える ----------
  private chooseGreedy(
    board: ReversiBoard,
    player: number,
    moves: [number, number][]
  ): [number, number] {
    let bestMove = moves[0];
    let bestScore = -Infinity;

    for (const [x, y] of moves) {
      const tmp = board.clone();
      tmp.applyMove(x, y, player);
      const { black, white } = tmp.countStones();
      const score = player === 1 ? black - white : white - black;
      if (score > bestScore) {
        bestScore = score;
        bestMove = [x, y];
      }
    }
    return bestMove;
  }

  // ---------- ミニマックス（簡易） ----------
  private chooseMinimax(
    board: ReversiBoard,
    player: number,
    moves: [number, number][]
  ): [number, number] {
    let bestMove = moves[0];
    let bestScore = -Infinity;

    for (const [x, y] of moves) {
      const tmp = board.clone();
      tmp.applyMove(x, y, player);
      const score = this.minimax(tmp, this.maxDepth - 1, -player, player);
      if (score > bestScore) {
        bestScore = score;
        bestMove = [x, y];
      }
    }
    return bestMove;
  }

  /**
   * minimax:
   * - currentPlayer: 今手を打つプレイヤー
   * - maximizingPlayer: 最終的にスコアを最大化したいプレイヤー（AI側）
   */
  private minimax(
    board: ReversiBoard,
    depth: number,
    currentPlayer: number,
    maximizingPlayer: number
  ): number {
    if (depth === 0 || board.isGameOver()) {
      return this.evaluate(board, maximizingPlayer);
    }

    const moves = board.getValidMoves(currentPlayer);

    // パスの場合（置ける場所が無いとき）は、相手に手番を渡す
    if (moves.length === 0) {
      return this.minimax(board, depth - 1, -currentPlayer, maximizingPlayer);
    }

    if (currentPlayer === maximizingPlayer) {
      // 最大化フェーズ
      let maxEval = -Infinity;
      for (const [x, y] of moves) {
        const tmp = board.clone();
        tmp.applyMove(x, y, currentPlayer);
        const evalScore = this.minimax(tmp, depth - 1, -currentPlayer, maximizingPlayer);
        if (evalScore > maxEval) {
          maxEval = evalScore;
        }
      }
      return maxEval;
    } else {
      // 最小化フェーズ（相手）
      let minEval = Infinity;
      for (const [x, y] of moves) {
        const tmp = board.clone();
        tmp.applyMove(x, y, currentPlayer);
        const evalScore = this.minimax(tmp, depth - 1, -currentPlayer, maximizingPlayer);
        if (evalScore < minEval) {
          minEval = evalScore;
        }
      }
      return minEval;
    }
  }

  // 評価関数：単純に石の差 + 角のボーナス
  private evaluate(board: ReversiBoard, player: number): number {
    const { black, white } = board.countStones();
    const stoneDiff = player === 1 ? black - white : white - black;

    // 角の評価（単純版）
    const corners: [number, number][] = [
      [0, 0],
      [0, board.size - 1],
      [board.size - 1, 0],
      [board.size - 1, board.size - 1],
    ];
    let cornerScore = 0;
    for (const [x, y] of corners) {
      if (board.board[y][x] === player) cornerScore += 5;
      if (board.board[y][x] === -player) cornerScore -= 5;
    }

    return stoneDiff * 1 + cornerScore;
  }
}

// ===== ここに ReversiBoard / ReversiAI の定義をコピペしておく =====

// ここからブラウザ連携コード
const humanPlayer = 1;   // 黒
const aiPlayer = -1;     // 白

let board = new ReversiBoard(8);
const ai = new ReversiAI("minimax", 3); // 強さはお好みで

const boardEl = document.getElementById("board") as HTMLDivElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;

function renderBoard() {
  boardEl.innerHTML = "";

  for (let y = 0; y < board.size; y++) {
    for (let x = 0; x < board.size; x++) {
      const cell = document.createElement("div");
      cell.classList.add("cell");
      cell.dataset.x = String(x);
      cell.dataset.y = String(y);

      const v = board.board[y][x];
      if (v === 1) cell.classList.add("black");
      if (v === -1) cell.classList.add("white");

      // 現在の手番プレイヤーの有効手をハイライト
      if (board.isValidMove(x, y, board.currentPlayer)) {
        cell.classList.add("valid");
      }

      boardEl.appendChild(cell);
    }
  }

  updateStatus();
}

function updateStatus() {
  const { black, white } = board.countStones();

  let msg = `黒: ${black} ／ 白: ${white}　`;
  if (board.isGameOver()) {
    if (black > white) msg += "→ 黒の勝ち！";
    else if (white > black) msg += "→ 白の勝ち！";
    else msg += "→ 引き分け";
  } else {
    const turnText =
      board.currentPlayer === humanPlayer ? "あなた（黒）の番です" : "AI（白）の思考中…";
    msg += `　手番: ${turnText}`;
  }

  statusEl.textContent = msg;
}

function humanClick(x: number, y: number) {
  if (board.currentPlayer !== humanPlayer) return;                    // 自分の手番以外は無視
  if (!board.isValidMove(x, y, humanPlayer)) return;                  // 非合法手なら無視
  board.applyMove(x, y, humanPlayer);
  renderBoard();

  if (board.isGameOver()) return;

  // 少し待ってからAIの手番
  setTimeout(aiTurn, 300);
}

function aiTurn() {
  if (board.currentPlayer !== aiPlayer) return;
  if (board.isGameOver()) return;

  const move = ai.getBestMove(board, aiPlayer);

  if (move) {
    const [x, y] = move;
    board.applyMove(x, y, aiPlayer);
  } else {
    // AIがパスの場合
    board.switchPlayer();
  }

  renderBoard();
}

// 盤面クリックイベント
boardEl.addEventListener("click", (e) => {
  const target = e.target as HTMLElement;
  if (!target.classList.contains("cell")) return;

  const x = Number(target.dataset.x);
  const y = Number(target.dataset.y);

  humanClick(x, y);
});

// リセットボタン
resetBtn.addEventListener("click", () => {
  board = new ReversiBoard(8);
  // 人間を先手にする
  board.currentPlayer = humanPlayer;
  renderBoard();
});

// 初期描画
renderBoard();
