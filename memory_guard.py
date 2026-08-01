# memory_guard.py
# 内存看门狗：独立线程每秒检测进程内存，超过阈值自动终止，防止系统死机
#
# 用法：
#   from memory_guard import start_guard
#   start_guard(MEMORY_LIMIT_GB)  # 在 train.py 开头调用
#
# 原理：
#   守护线程每秒读取 psutil.Process().memory_info().rss
#   rss（Resident Set Size）是进程实际占用的物理内存
#   超限时调用 os._exit(1) 强制终止，不给 GC 和换页留时间

import os
import sys
import time
import threading

_guard = None  # 全局实例引用


class MemoryGuard:
    def __init__(self, limit_gb, check_interval=1.0):
        self.limit_bytes = int(limit_gb * 1024**3)
        self.check_interval = check_interval
        self._thread = None
        self._stop_event = threading.Event()

    def _get_rss_bytes(self):
        """获取当前进程物理内存占用（字节）"""
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss
        except ImportError:
            # psutil 不可用时下载并重试
            print("[MemoryGuard] psutil 未安装，尝试安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "-q"])
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss

    def _watch(self):
        """守护线程主循环"""
        peak_rss = 0
        last_warn = 0
        try:
            while not self._stop_event.is_set():
                rss = self._get_rss_bytes()
                peak_rss = max(peak_rss, rss)
                rss_gb = rss / 1024**3
                limit_gb = self.limit_bytes / 1024**3

                # 在 80% 阈值时发一次警告
                if rss >= self.limit_bytes * 0.8 and rss < self.limit_bytes:
                    if time.time() - last_warn > 10:  # 每 10 秒最多警告一次
                        print(f"\n  ⚠️ [MemoryGuard] 内存 {rss_gb:.1f} GB，已使用 {rss_gb/limit_gb*100:.0f}% 阈值 ({limit_gb:.0f} GB)")
                        last_warn = time.time()

                # 超限：立即终止
                if rss >= self.limit_bytes:
                    print(f"\n{'='*60}")
                    print(f"  🛑 [MemoryGuard] 进程内存 {rss_gb:.1f} GB 超过限制 {limit_gb:.0f} GB！")
                    print(f"  峰值内存: {peak_rss/1024**3:.1f} GB")
                    print(f"  正在强制终止进程以防止系统死机...")
                    print(f"{'='*60}")
                    os._exit(1)  # 强制终止，不等待清理

                time.sleep(self.check_interval)
        except Exception as e:
            print(f"\n  [MemoryGuard] 监控异常: {e}")
            os._exit(1)

    def start(self):
        """启动守护线程"""
        if self._thread is not None:
            return
        limit_gb = self.limit_bytes / 1024**3
        print(f"  🛡️ [MemoryGuard] 已启动 (限制: {limit_gb:.0f} GB, 检查间隔: {self.check_interval}s)")
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        """停止守护线程"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def start_guard(limit_gb, check_interval=1.0):
    """快捷启动内存看门狗"""
    global _guard
    if _guard is None:
        _guard = MemoryGuard(limit_gb, check_interval)
        _guard.start()
    return _guard


def stop_guard():
    """停止内存看门狗"""
    global _guard
    if _guard is not None:
        _guard.stop()
        _guard = None