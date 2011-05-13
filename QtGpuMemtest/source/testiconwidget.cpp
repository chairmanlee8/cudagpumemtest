#include "testiconwidget.h"
#include <algorithm>

TestIconWidget::TestIconWidget(TestInfo& aTestInfo, QWidget *parent)
	: QWidget(parent), testStatus(TestNotStarted), widgetMode(SelectMode)
{
	// setup this widget's resizing behavior
	setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	
	// setup the test info
	setTestInfo(aTestInfo);

	mouseClicked = false;
	hover = false;

	// setup the flasher
	flashTimer = new QTimer(this);
	flashColor = QColor(150, 150, 150);

	connect(flashTimer, SIGNAL(timeout()), this, SLOT(updateRunningColor()));
}

TestIconWidget::~TestIconWidget()
{
}

void TestIconWidget::setTestInfo(TestInfo& anotherTestInfo)
{
	testInfo = anotherTestInfo;

	this->setToolTip(testInfo.testName);
	// Calculate width
	QFontMetrics fm(font());
	adjWidth = std::max(26, 20 + fm.width(testInfo.testShortName));

	update();
}

void TestIconWidget::setStatus(TestStatus aStatus)
{
	testStatus = aStatus;

	if(testStatus == TestRunning)
	{
		// enable animation timer callback
		flashTimer->start(50);
	}
	else
	{
		// disable timer callback
		flashTimer->stop();
	}

	update();
}

void TestIconWidget::setState(Mode m)
{
	widgetMode = m;
	update();
}

QSize TestIconWidget::sizeHint() const
{
	return QSize(adjWidth, 26);
}

void TestIconWidget::updateRunningColor()
{
	static int direction = 1;

	// Update the running color
	flashColor.setGreen(flashColor.green() + 10 * direction);

	if(flashColor.green() >= 245) direction = -1;
	else if (flashColor.green() < 150) direction = 1;

	update();
}

void TestIconWidget::paintEvent(QPaintEvent* event)
{
	QPainter p(this);

	QSize frame = this->frameSize();
	QRect frameRect = QRect(0, 0, frame.width(), frame.height());

	// Initialize our palette and drawing areas
	QBrush grayBrush(QColor(150, 150, 150), Qt::SolidPattern);
	QBrush darkGrayBrush(QColor(64, 64, 64), Qt::SolidPattern);
	QBrush greenBrush(QColor(0, 128, 0), Qt::SolidPattern);
	QBrush redBrush(QColor(192, 0, 0), Qt::SolidPattern);
	QPen whitePen(QColor(255, 255, 255));

	// Fill the background based on the mode and status
	if(widgetMode == DisplayMode)
	{
		switch(testStatus)
		{
		case TestNotStarted:
			p.fillRect(frameRect, grayBrush);
			break;
		case TestPassed:
			p.fillRect(frameRect, greenBrush);
			break;
		case TestFailed:
			p.fillRect(frameRect, redBrush);
			break;
		case TestRunning:
			p.fillRect(frameRect, QBrush(flashColor, Qt::SolidPattern));
			break;
		}

		p.setPen(whitePen);
		p.drawText(frameRect, Qt::AlignCenter, testInfo.testShortName);
	}
	else// widgetMode == Mode::SelectMode
	{
		// Select mode, fill gray then draw test symbol
		// Draw border green if test is enabled
		p.setBrush(grayBrush);
		p.fillRect(frameRect, grayBrush);

		if(testInfo.testEnabled)
		{
			p.setPen(QPen(greenBrush, 6));
			p.drawRect(frameRect);
		}
		else
		{
			if(hover)
			{
				p.setPen(QPen(darkGrayBrush, 6));
				p.drawRect(frameRect);
			}
		}

		p.setPen(whitePen);
		p.drawText(frameRect, Qt::AlignCenter, testInfo.testShortName);
	}
}

void TestIconWidget::mousePressEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton)
	{
		mouseClicked = true;
	}
}

void TestIconWidget::mouseReleaseEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton)
	{
		if(mouseClicked)
		{
			// registered a mouse click, handle it
			if(widgetMode == SelectMode)
			{
				testInfo.testEnabled = !testInfo.testEnabled;
				update();
			}
		}

		mouseClicked = false;
	}
}

void TestIconWidget::mouseDoubleClickEvent(QMouseEvent* event)
{
	// If double click is on a solo-ed item, select all
	// If double click is on a non-solo selected item, solo it
	// If double click is on a non-selected item, select all except it (is this intuitive?)

	testInfo.testEnabled = !testInfo.testEnabled;
	emit maskEnable(testInfo, testInfo.testEnabled);

	update();
}

void TestIconWidget::enterEvent(QEvent* event)
{
	hover = true;
	update();
}

void TestIconWidget::leaveEvent(QEvent* event)
{
	hover = false;
	update();
}