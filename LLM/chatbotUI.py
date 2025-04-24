import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QListView, 
    QMessageBox, QVBoxLayout, QStyledItemDelegate
)
from PyQt5.QtCore import (
    Qt, QAbstractListModel, QMargins, QSize, QRect, QPoint, QThread, pyqtSignal
)
from PyQt5.QtGui import QIcon, QColor, QImage, QPolygon, QFontMetrics

# Import chatbotGPT to use its functionality (code 2)
import chatbotGPT

# Style sheet for the chat window elements
style_sheet = """ 
    QListView {
        background: #FDF3DD;
    }
"""

class ChatLogModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.chat_messages = []

    def rowCount(self, index):
        return len(self.chat_messages)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.chat_messages[index.row()]

    def appendMessage(self, message, sender):
        """
        Append a new message to the chat log.
        sender should be either "user" or "chatbot".
        """
        self.chat_messages.append([message, sender])
        self.layoutChanged.emit()

class DrawSpeechBubbleDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_offset = 5  # Horizontal offset for the image
        # Offsets for drawing the speech bubble and its tail
        self.side_offset, self.top_offset = 40, 5
        self.tail_offset_x, self.tail_offset_y = 30, 0
        self.text_side_offset, self.text_top_offset = 50, 15

    def paint(self, painter, option, index):
        """
        Custom paint method to draw each chat message as a speech bubble
        with an icon.
        """
        text, sender = index.model().data(index, Qt.DisplayRole)
        image, image_rect = QImage(), QRect()
        color, bubble_margins = QColor(), QMargins()
        tail_points = QPolygon()

        if sender == "chatbot":
            image.load("bot.png")
            image_rect = QRect(
                QPoint(option.rect.left() + self.image_offset, option.rect.center().y() - 12),
                QSize(24, 24)
            )
            color = QColor("#83E56C")
            bubble_margins = QMargins(self.side_offset, self.top_offset, self.side_offset, self.top_offset)
            tail_points = QPolygon([
                QPoint(option.rect.x() + self.tail_offset_x, option.rect.center().y()),
                QPoint(option.rect.x() + self.side_offset, option.rect.center().y() - 5),
                QPoint(option.rect.x() + self.side_offset, option.rect.center().y() + 5)
            ])
        elif sender == "user":
            image.load("user.png")
            image_rect = QRect(
                QPoint(option.rect.right() - self.image_offset - 24, option.rect.center().y() - 12),
                QSize(24, 24)
            )
            color = QColor("#38E0F9")
            bubble_margins = QMargins(self.side_offset, self.top_offset, self.side_offset, self.top_offset)
            tail_points = QPolygon([
                QPoint(option.rect.right() - self.tail_offset_x, option.rect.center().y()),
                QPoint(option.rect.right() - self.side_offset, option.rect.center().y() - 5),
                QPoint(option.rect.right() - self.side_offset, option.rect.center().y() + 5)
            ])

        # Draw the icon
        painter.drawImage(image_rect, image)
        # Draw the speech bubble
        painter.setPen(color)
        painter.setBrush(color)
        painter.drawRoundedRect(option.rect.marginsRemoved(bubble_margins), 5, 5)
        painter.drawPolygon(tail_points)
        # Draw the text inside the speech bubble
        painter.setPen(QColor("#4A4C4B"))
        text_margins = QMargins(self.text_side_offset, self.text_top_offset,
                                self.text_side_offset, self.text_top_offset)
        painter.drawText(option.rect.marginsRemoved(text_margins),
                         Qt.AlignVCenter | Qt.TextWordWrap,
                         text)

    def sizeHint(self, option, index):
        text, sender = index.model().data(index, Qt.DisplayRole)
        font_metrics = QFontMetrics(option.font)
        max_bubble_width = 250
        text_rect = font_metrics.boundingRect(0, 0, max_bubble_width, 0,
                                              Qt.TextWordWrap, text)
        text_rect = text_rect.marginsAdded(QMargins(self.text_side_offset,
                                                    self.text_top_offset,
                                                    self.text_side_offset,
                                                    self.text_top_offset))
        return text_rect.size()

class ChatbotWorker(QThread):
    resultReady = pyqtSignal(str)
    
    def __init__(self, user_input, sql_history, db_connection, parent=None):
        super().__init__(parent)
        self.user_input = user_input
        self.sql_history = sql_history
        self.db_connection = db_connection

    def run(self):
        # Append user message to conversation history
        self.sql_history.append({"role": "user", "content": self.user_input})
        # Generate SQL query using chatbotGPT's ollama_send_message
        raw_sql_response = chatbotGPT.ollama_send_message(self.sql_history, chatbotGPT.SQL_MODEL_NAME)
        extracted_sql = chatbotGPT.extract_sql_from_response(raw_sql_response)
        if not extracted_sql:
            self.resultReady.emit("I am AI agent ")
            return
        self.sql_history.append({"role": "assistant", "content": extracted_sql})
        # Execute the SQL query and analyze results
        sql_result = chatbotGPT.execute_sql_query(self.db_connection, extracted_sql)
        final_answer = chatbotGPT.analyze_results_with_llm(self.user_input, extracted_sql, sql_result)
        self.resultReady.emit(final_answer)

class Chatbot(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()
        # Initialize conversation history using the SQL system prompt from chatbotGPT (do not change the prompts)
        self.sql_system_prompt = (
            "You are a specialized Text-to-SQL assistant for MySQL. Your ONLY task is to convert natural language questions into VALID and EXECUTABLE MySQL SQL queries based on the provided table context.\n\n"
            "VERY STRICT INSTRUCTIONS:\n"
            "1. OUTPUT ONLY THE SQL QUERY. Generate nothing else.\n"
            "2. ENSURE MYSQL SYNTAX. Use backticks for identifiers if needed and single quotes for strings.\n"
            "3. TERMINATE WITH SEMICOLON.\n"
            "4. HANDLE AMBIGUITY appropriately.\n"
            "5. IGNORE NON-SQL REQUESTS.\n"
            "6. NO SCHEMA MODIFICATION.\n"
            "7. USE PROVIDED CONTEXT ONLY.\n"
            "8. PAY ATTENTION TO FORMATS.\n"
            "\n**Table Context:**\n" + chatbotGPT.TABLE_CONTEXT
        )
        self.sql_conversation_history = [{"role": "system", "content": self.sql_system_prompt}]
        # Create a DB connection using chatbotGPT functions
        self.db_connection = chatbotGPT.connect_to_db(chatbotGPT.DB_CONFIG)

    def initializeUI(self):
        """
        Initialize the chat window and its widgets.
        """
        self.setFixedSize(450, 600)
        self.setWindowTitle("Chatbot")
        self.setWindowFlag(Qt.Window)
        self.setupWindow()
        self.model.appendMessage("Hey!!!.", "chatbot")
        self.user_input_line.setPlaceholderText("Type your message and press 'Enter'")
        self.show()

    def setupWindow(self):
        """
        Set up the UI components, including the chat log and input field.
        """
        self.model = ChatLogModel()
        self.chat_log_view = QListView()
        self.chat_log_view.setModel(self.model)
        delegate = DrawSpeechBubbleDelegate(parent=self.chat_log_view)
        self.chat_log_view.setItemDelegate(delegate)
        self.chat_log_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.user_input_line = QLineEdit()
        self.user_input_line.setMinimumHeight(24)
        self.user_input_line.returnPressed.connect(self.enterUserMessage)
        main_v_box = QVBoxLayout()
        main_v_box.setContentsMargins(0, 2, 0, 10)
        main_v_box.setSpacing(10)
        main_v_box.addWidget(self.chat_log_view)
        main_v_box.addWidget(self.user_input_line)
        self.setLayout(main_v_box)

    def enterUserMessage(self):
        user_input = self.user_input_line.text().strip()
        if user_input:
            # Append user's message to the chat log
            self.model.appendMessage(user_input, "user")
            self.user_input_line.clear()
            
            # Append temporary "Working on it..." message
            self.model.appendMessage("Working on it...", "chatbot")
            QApplication.processEvents()  # Ensure UI updates
            
            # Start worker thread to process LLM call
            self.worker = ChatbotWorker(user_input, self.sql_conversation_history, self.db_connection)
            self.worker.resultReady.connect(self.handleResult)
            self.worker.start()

    def handleResult(self, final_answer):
        # Remove the temporary "Working on it..." message (assumed to be the last one)
        if self.model.chat_messages and self.model.chat_messages[-1][0] == "Working on it...":
            self.model.chat_messages.pop()
            self.model.layoutChanged.emit()
        # Append the final response from the LLM
        self.model.appendMessage(final_answer, "chatbot")

    def closeEvent(self, event):
        """
        Confirm with the user before closing the chat window.
        """
        choice = QMessageBox.question(
            self, 'Leave Chat?', "Are you sure you want to leave the chat?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if choice == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheet)
    window = Chatbot()
    sys.exit(app.exec_())
