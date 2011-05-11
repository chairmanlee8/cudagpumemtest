#include "gpudisplaywidget.h"
#include "qtgpumemtest.h"

GpuDisplayWidget::GpuDisplayWidget(QWidget *parent)
	: QWidget(parent), m_widgetIndex(0), gpuThread(0)
{
	layout = new QVBoxLayout(this);
	innerTopLayout = new QHBoxLayout();
	innerBottomLayout = new QHBoxLayout();

	startButton = new QToolButton();
	startLoopButton = new QToolButton();
	startStressButton = new QToolButton();
	stopButton = new QToolButton();
	resultsButton = new QToolButton();

	startButton->setIcon(QIcon("play_18x24_blue.png"));
	startButton->setIconSize(QSize(24, 24));
	startLoopButton->setIcon(QIcon("playinfinite_18x24_blue.png"));
	startLoopButton->setIconSize(QSize(24, 24));
	stopButton->setIcon(QIcon("stop_16x16_blue.png"));
	stopButton->setIconSize(QSize(24, 24));
	stopButton->setEnabled(false);
	resultsButton->setIcon(QIcon("book_24x24.png"));
	resultsButton->setIconSize(QSize(24, 24));

	m_labelGpu = new QLabel("Default GPU");
	m_labelMemory = new QLabel("#Memory");
	m_progress = new QHBoxLayout();
	m_progress->setAlignment(Qt::AlignLeft);
	m_checkStart = new QCheckBox();
	m_labelStopping = new QLabel();

	m_progress->setSpacing(1);
	m_labelGpu->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	innerTopLayout->addWidget(m_checkStart);
	innerTopLayout->addWidget(m_labelGpu);
	layout->addLayout(innerTopLayout);
	layout->addWidget(m_labelMemory);
	layout->addLayout(m_progress);
	innerBottomLayout->addWidget(startButton);
	innerBottomLayout->addWidget(startLoopButton);
	innerBottomLayout->addWidget(stopButton);
	innerBottomLayout->addWidget(m_labelStopping);
	innerBottomLayout->addWidget(resultsButton, 0, Qt::AlignRight);
	layout->addLayout(innerBottomLayout);

	this->setStyleSheet
	(
		"GpuDisplayWidget { \
		background-color: qradialgradient(cx: 0, cy: 0, fx: 1, fy: 1, radius: 1, stop: 0 #ffffff, stop: 1 #8BDEFF); \
			margin-top: 1px; \
			margin-bottom: 1px; \
			border: 2px solid #333333; \
			border-radius: 4px \
		} \
		QToolButton { \
			border: none; \
		}"
	);

	// Connect signals and slots
	connect(startButton, SIGNAL(clicked()), this, SLOT(startTestOnce()));
	connect(startLoopButton, SIGNAL(clicked()), this, SLOT(startTestInfinite()));
	connect(stopButton, SIGNAL(clicked()), this, SLOT(endTest()));
	connect(resultsButton, SIGNAL(clicked()), this, SLOT(displayLog()));
}

GpuDisplayWidget::~GpuDisplayWidget()
{

}

void GpuDisplayWidget::startTestOnce()
{
	startTest(false);
}

void GpuDisplayWidget::startTestInfinite()
{
	startTest(true);
}

void GpuDisplayWidget::startTest(bool infinite)
{
	// Blah blah load test options and what not, configuration and stuff, anything that needs to be
	// passed to the parent window/process.
	emit testStarted(m_widgetIndex, infinite);

	startButton->setEnabled(false);
	startLoopButton->setEnabled(false);
	stopButton->setEnabled(true);

	// Load individual test options (tests enabled/disabled)
	QVector<TestInfo> newTests;
	for(int i = 0; i < testWidgets.size(); i++)
	{
		testWidgets[i]->setMode(TestIconWidget::DisplayMode);
		newTests.push_back(testWidgets[i]->getTestInfo());
	}

	// Setup GPU thread
	gpuThread = new QtGpuThread(newTests);

	connect(gpuThread, SIGNAL(failed(int, QString)), this, SLOT(testFailed(int, QString)));
	connect(gpuThread, SIGNAL(passed(int, QString)), this, SLOT(testPassed(int, QString)));
	connect(gpuThread, SIGNAL(starting(int, QString)), this, SLOT(testStarting(int, QString)));
	connect(gpuThread, SIGNAL(finished()), this, SLOT(endTest()));
	connect(gpuThread, SIGNAL(log(int, QString, QString)), this, SLOT(testLog(int, QString, QString)));

	// Link the controller to test ended signals
	// Notify the controller of how many tests will be performed for this widget
	int enabledCount = 0;
	for(int i = 0; i < newTests.count(); i++)
	{
		if(newTests[i].testEnabled) enabledCount++;
	}
	m_controller->testsStarted(enabledCount);
	connect(gpuThread, SIGNAL(ended(int, QString)), m_controller, SLOT(testEnded(int, QString)));

	gpuThread->setDevice(m_widgetIndex);
	if(infinite) gpuThread->setEndless(true);
	gpuThread->start();
}

void GpuDisplayWidget::endTest()
{
	static unsigned int timesEnded = 0;

	if(!stopButton->isEnabled()) return;	// Nothing to stop

	if(timesEnded == 0)
	{
		// Is the thread already stopped? (naturally stopped)
		// then call endTest again
		if(!gpuThread->isRunning())
		{
			timesEnded += 2;
			m_labelStopping->setText(QString("Done, click stop to reset."));
		}
		else
		{
			// Kill thread
			gpuThread->notifyExit();
			m_labelStopping->setText(QString("Stopping..."));
			timesEnded++;
		}
	}
	else if(timesEnded == 1)
	{
		// Do nothing if gpuThread is still running, otherwise increment timesEnded
		if(!gpuThread->isRunning())
		{
			emit testEnded(m_widgetIndex);

			m_labelStopping->setText(QString("Done, click stop again to reset."));
			timesEnded++;
		}
	}
	else if(timesEnded > 1)
	{
		m_labelStopping->setText(QString(""));

		for(int i = 0; i < testWidgets.size(); i++)
		{
			testWidgets[i]->setMode(TestIconWidget::SelectMode);
			testWidgets[i]->setStatus(TestNotStarted);
		}

		startButton->setEnabled(true);
		startLoopButton->setEnabled(true);
		stopButton->setEnabled(false);
		timesEnded = 0;
	}
}

void GpuDisplayWidget::displayLog()
{
	QString wtitle;
	QTextStream t(&wtitle);

	t << "Results [" << m_labelGpu->text() << "]";

	ResultsDisplay *rd = new ResultsDisplay();
	rd->setResults(log);
	rd->setWindowTitle(wtitle);
	rd->show();
}

QString GpuDisplayWidget::getLog()
{
	QString t;
	QTextStream tt(&t);

	for(int i = 0; i < log.size(); i++)
	{
		tt << log[i] << "\r\n";	// TODO: UNIX or CR/LF line endings
	}

	return t;
}

void GpuDisplayWidget::testFailed(int deviceIdx, QString testName)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testName == testWidgets[i]->getTestInfo().testName)
		{
			testWidgets[i]->setStatus(TestFailed);
			break;
		}
	}
}

void GpuDisplayWidget::testPassed(int deviceIdx, QString testName)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testName == testWidgets[i]->getTestInfo().testName)
		{
			testWidgets[i]->setStatus(TestPassed);
			break;
		}
	}
}

void GpuDisplayWidget::testStarting(int deviceIdx, QString testName)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testName == testWidgets[i]->getTestInfo().testName)
		{
			testWidgets[i]->setStatus(TestRunning);
			break;
		}
	}
}

void GpuDisplayWidget::testLog(int deviceIdx, QString testName, QString logMessage)
{
	QString logString("");
	QTextStream logStream(&logString);

	logStream << "[" << QTime::currentTime().toString() << "][Device " << deviceIdx << "](" << testName << ") " << logMessage;
	logStream.flush();

	log.push_back(logString);
}

void GpuDisplayWidget::paintEvent(QPaintEvent* event)
{
	QStyleOption opt;
	opt.init(this);
	QPainter p(this);
	style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
	//QWidget::paintEvent(event);
}

QFont GpuDisplayWidget::font() const
{
	return m_labelGpu->font();
}

void GpuDisplayWidget::setFont(const QFont &font)
{
	m_labelGpu->setFont(font);

	QFont *newFont = new QFont(font);
	newFont->setPointSize(newFont->pointSize() - 4);

	m_labelMemory->setFont(*newFont);
}

int GpuDisplayWidget::index() const
{
	return m_widgetIndex;
}

void GpuDisplayWidget::setIndex(int idx)
{
	m_widgetIndex = idx;
}

void GpuDisplayWidget::setCheckStart(const int checked)
{
	m_checkStart->setChecked(checked > 0);
}

void GpuDisplayWidget::setGpuName(const QString& gpuName)
{
	m_labelGpu->setText(gpuName);
}

void GpuDisplayWidget::setGpuMemory(const QString& gpuMemory)
{
	m_labelMemory->setText(gpuMemory);
}

void GpuDisplayWidget::setTestStatus(const TestStatus& gpuTestStatus)
{
}

void GpuDisplayWidget::setTests(QVector<TestInfo>& aTests)
{
	// Remove old tests and add new ones
	for(int i = 0; i < tests.size(); i++)
	{
		m_progress->removeWidget(testWidgets[i]);
		delete testWidgets[i];
	}

	tests.clear();
	testWidgets.clear();

	for(int i = 0; i < aTests.size(); i++)
	{
		tests.push_back(aTests[i]);

		TestIconWidget* aWidget = new TestIconWidget(tests[i]);
		//aWidget->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		
		if(aTests[i].testShortName.length() > 1)
		{
			// TODO: actually calculate the width of the widget
			aWidget->setWidth(56);
		}

		m_progress->addWidget(aWidget);

		testWidgets.push_back(aWidget);
	}
}

bool GpuDisplayWidget::isTestFailed()
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testWidgets[i]->getStatus() == TestFailed)
			return true;
	}

	return false;
}