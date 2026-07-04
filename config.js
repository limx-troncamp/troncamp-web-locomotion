// 合并榜单数据源配置（单 board：TronCamp Task C + Humanoid Task F 同站同榜）。
// 两份数据都在本 board 仓库里、各自独立：tron 走 ./data/leaderboard.json，人形走 ./data/humanoid.json。
// 默认读同站静态文件（由 Worker push 到本 Pages 仓库 data/ 下）。
// 机型权重(wfyg 轮式 ×0.8)已在 Worker 端算进每行 total；前端只按 total 排名、不再二次相乘。
window.BOARD_CONFIG = {
  TRON_DATA_URL:     "./data/leaderboard.json",
  HUMANOID_DATA_URL: "./data/humanoid.json",
  REFRESH_SECONDS:   60,
  KIT_BASE:          "./participant_kit/",   // 资源页：participant_kit 在站点的挂载路径
};
