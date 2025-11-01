# -*- coding: utf-8 -*-
"""集中存放界面样式，避免主界面模块过长。"""

APP_STYLE = """
/* 全局 */
QWidget {
  background-color: #F5F6FA;
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI";
  font-size: 14px;
  color: #1F1F1F;
}

/* 顶部标题栏 */
#TitleBar {
  background: #F5F6FA;
  border-bottom: 1px solid #E5E7EB;
}
#pageTitle {
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI";
  font-size: 18px;
  font-weight: 700;
  color: #111827;
}

/* 卡片/分组 */
QGroupBox {
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  margin-top: 12px;
  background: #FFFFFF;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 6px;
  font-weight: 600;
  color: #111827;
}

/* 输入控件 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: 8px 10px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
  border: 1px solid #60A5FA;
}

/* 主按钮（执行类） */
QPushButton {
  background-color: #0EA5E9;
  color: #FFFFFF;
  border-radius: 10px;
  padding: 8px 16px;
  min-height: 38px;
  border: 0;
}
QPushButton:hover { background-color: #38BDF8; }
QPushButton:pressed { background-color: #0284C7; }
QPushButton:disabled { background-color: #C7CDD4; color: #F9FAFB; }

/* 次要按钮/分页按钮 */
QPushButton#secondary {
  background: #EEF1F6;
  color: #111827;
  border-radius: 10px;
  padding: 8px 16px;
  min-height: 38px;
}
QPushButton#secondary:hover { background: #E5EAF1; }
QPushButton#secondary:pressed { background: #D9E0EA; }

/* 表格 */
QHeaderView::section {
  background: #F3F4F6;
  border: 1px solid #E5E7EB;
  padding: 8px;
  font-weight: 600;
}
QTableView {
  background: #FFFFFF;
  gridline-color: #E5E7EB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

/* 分隔条 */
QSplitter::handle {
  background-color: #E5E7EB;
  width: 2px;
}

/* 状态栏 */
QStatusBar {
  background: #F3F4F6;
  border-top: 1px solid #E5E7EB;
  font-size: 12px;
  color: #6B7280;
  padding: 4px 8px;
}

/* 列表/文本区域 */
QTextEdit, QPlainTextEdit, QListWidget {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

/* 工具按钮 & 滚动区 */
QToolButton {
  background: transparent;
  border: none;
  padding: 4px;
}
QToolButton:hover {
  background: rgba(14, 165, 233, 0.1);
  border-radius: 6px;
}
QScrollArea {
  border: none;
}

/* 在线状态徽标 */
#OnlineStatusLabel {
  font-size: 12px;
  color: #5F6368;
  padding: 6px 8px;
  border-radius: 6px;
  background-color: #F3F4F6;
}
"""
