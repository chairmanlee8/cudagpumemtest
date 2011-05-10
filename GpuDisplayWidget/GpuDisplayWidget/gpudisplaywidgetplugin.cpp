#include "gpudisplaywidget.h"

#include <QtCore/QtPlugin>
#include "gpudisplaywidgetplugin.h"


GpuDisplayWidgetPlugin::GpuDisplayWidgetPlugin(QObject *parent)
	: QObject(parent)
{
	initialized = false;
}

void GpuDisplayWidgetPlugin::initialize(QDesignerFormEditorInterface * /*core*/)
{
	if (initialized)
		return;

	initialized = true;
}

bool GpuDisplayWidgetPlugin::isInitialized() const
{
	return initialized;
}

QWidget *GpuDisplayWidgetPlugin::createWidget(QWidget *parent)
{
	return new GpuDisplayWidget(parent);
}

QString GpuDisplayWidgetPlugin::name() const
{
	return "GpuDisplayWidget";
}

QString GpuDisplayWidgetPlugin::group() const
{
	return "My Plugins";
}

QIcon GpuDisplayWidgetPlugin::icon() const
{
	return QIcon();
}

QString GpuDisplayWidgetPlugin::toolTip() const
{
	return QString();
}

QString GpuDisplayWidgetPlugin::whatsThis() const
{
	return QString();
}

bool GpuDisplayWidgetPlugin::isContainer() const
{
	return false;
}

QString GpuDisplayWidgetPlugin::domXml() const
{
	return "<widget class=\"GpuDisplayWidget\" name=\"gpuDisplayWidget\">\n"
		" <property name=\"geometry\">\n"
		"  <rect>\n"
		"   <x>0</x>\n"
		"   <y>0</y>\n"
		"   <width>100</width>\n"
		"   <height>100</height>\n"
		"  </rect>\n"
		" </property>\n"
		"</widget>\n";
}

QString GpuDisplayWidgetPlugin::includeFile() const
{
	return "gpudisplaywidget.h";
}

Q_EXPORT_PLUGIN2(gpudisplaywidget, GpuDisplayWidgetPlugin)
