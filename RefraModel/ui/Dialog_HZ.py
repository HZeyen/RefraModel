# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:41:31 2024

@author: Hermann
"""

from PyQt5 import QtWidgets, QtCore


class Dialog(QtWidgets.QWidget):
    """
    Created on Tue Sep 13 12:50:26 2022
    @author: Hermann
    General purpose dialog box

    Input:
        parent (Class object):
            Certain values are needed from the parent class usually, the call
            will thus be:
                Dialog(self,labels,types,values,title)
        labels (list, string): labels explaining the different fields
            Explanatory text put beside the fields for data entrance.
            If values==None, label is interpreted as information/Warning text.
            For a series of Radiobuttons or Comboboxes, all corresponding
            labels are given as a list within the list. See descriptionof
            "types" for specific formats.
        types (list,string): In the order of "labels", indicates the type
            of field the label belongs to.  If labels[i] is itself  a list,
            this is considered as one single label. May have the following
            values:
            - "c": to define a check box;
            - "e": to define an editable box (lineEdit);
            - "l": To define a simple label, comment text;
            - "r": to define a series of radiobuttons (the corresponding list
              of labels is considered as one single label for the numbering of
              labels, types and values);
            - "b": to define a combobox (dropdown menu).

            For 'l', 'e' and 'b', capitalization does not matter. However for
            'c' and 'r' it does:
            For these two types, if capital letter is used, this means that the
            fields change when the corresponding button is clicked. Usually,
            one or more widgets will be added to the dialog box or removed from
            it. If this is the case, the label(s) (and type(s) and value(s)) of
            the optional fields must follow immediately the definition of the
            check box/radio button and the label must have the following form:
            @ntext. "@" is the sign for "optional", n is the value of the
            controlling button (may be several ciffers). For check box, n=0
            means that the optional field is activated if the combo box is
            deactivated, if n=1, the field is activated if the box is
            activated. For radio buttons, n is the number of the radio button
            (starting at 0) which must be checked for the optional field to be
            activated.
            Example:
                label="check", type="C", value=0
                label="@1optional1", type="e", value=100.
                label="@1optional2", type="e", value=-1.
                label="@0no entry", type="l", value=None
            In this case, if the check box is checked, two lineEdit boxes will
            be shown with labels "optional1" and "optional2" and default values
            100. and -1. If the check box is unchecked, a text line will be
            shown "no entry". It is, however, not necessary to give a field for
            both cases (checked and unchecked). Usually, the optional
            fields will only be shown after checking a box.
        values (list, string, float or int): initial values for LineEdit
            - Initial values for LineEdit fields (float, int or str)
            - None for comboboxes
            - >0 for check box if it should be checked from the beginning,
              0 else.
            - Number of radiobutton to be activated by default (natural
              numbering, starting at 1, not at 0).
            - For labels, it may be "b" (bold text), "i" (italic) or anything
              else, including None, for standard text.
        title (string): Title of the dialog window
                            Optional, default value: "Title"
    """
# Create window

    def __init__(self, labels, types, values=None, title="Title"):
        import numpy as np
        super(Dialog, self).__init__()
        nlab = len(labels)
        self.labels = labels
        self.types = types
        self.values = values
        self.ck_order = np.zeros(nlab, dtype=int)
        self.n_checked = 0
        self.Dfinish = False
        self.Dbutton = False
        self.dlabels = []
        self.dlines = []
        self.ckState = []
        self.ckb = []
        self.rbtn = []
        self.btngroup = []
        self.combo = []
        self.radio = False
        self.check = False
        nlab = len(self.types)
        for i in range(nlab):
            self.ckState.append(False)
        self.YesBtn = QtWidgets.QPushButton('Ok', self)
        self.YesBtn.move(10, 20*(nlab+3))
        self.CancelBtn = QtWidgets.QPushButton('Cancel', self)
        self.CancelBtn.move(150, 20*(nlab+3))
        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayout(self.mainLayout)
        il_add = 0
        ilin = 0
        for i, t in enumerate(self.types):
            il = ilin + il_add
            if t.lower() == 'l':
                if self.values[i]:
                    if self.values[i].lower() == 'b':
                        self.dlabels.append(QtWidgets.QLabel
                                            ("<b>"+self.labels[i]+"</b>"))
                    elif self.values[i].lower() == 'i':
                        self.dlabels.append(QtWidgets.QLabel
                                            ("<i>"+self.labels[i]+"</i>"))
                    else:
                        self.dlabels.append(QtWidgets.QLabel(self.labels[i]))
                else:
                    self.dlabels.append(QtWidgets.QLabel(self.labels[i]))
                self.dlines.append(None)
                self.ckb.append(None)
                self.rbtn.append(None)
                self.btngroup.append(None)
                self.combo.append(None)
                self.mainLayout.addWidget(self.dlabels[ilin], il, 0, 1, 2)
                ilin += 1
            elif t.lower() == 'e':
                self.dlabels.append(QtWidgets.QLabel(self.labels[i]))
                self.dlines.append(QtWidgets.QLineEdit())
                self.ckb.append(None)
                self.rbtn.append(None)
                self.btngroup.append(None)
                self.combo.append(None)
                self.mainLayout.addWidget(self.dlabels[ilin], il, 0, 1, 1)
                self.mainLayout.addWidget(self.dlines[ilin], il, 1, 1, 1)
                try:
                    s = str(self.values[i])
                    self.dlines[-1].setText(s)
                except:
                    pass
                ilin += 1
            elif t.lower() == 'r':
                self.ckb.append(None)
                self.combo.append(None)
                self.rbtn.append([])
                self.btngroup.append(QtWidgets.QButtonGroup())
                rck = int(self.values[i])-1
                if rck < 0 or rck >= len(self.labels[i]):
                    rck = 0
                for ir, lab in enumerate(self.labels[i]):
                    self.dlabels.append(None)
                    self.dlines.append(None)
                    self.rbtn[i].append(QtWidgets.QRadioButton(lab))
                    self.rbtn[i][-1].setProperty("groupName", str(i))
                    self.btngroup[-1].addButton(self.rbtn[i][-1])
                    self.mainLayout.addWidget(self.rbtn[i][-1], il, 0, 1, 2)
                    if ir == rck:
                        self.rbtn[i][-1].setChecked(True)
                    else:
                        self.rbtn[i][-1].setChecked(False)
                    il += 1
                    ilin += 1
                if t == "R":
                    self.btngroup[-1].buttonClicked.connect(self.radio_checked)
            elif t.lower() == 'c':
                self.dlabels.append(None)
                self.dlines.append(None)
                self.rbtn.append(None)
                self.btngroup.append(None)
                self.combo.append(None)
                self.ckb.append(QtWidgets.QCheckBox(self))
                self.ckb[i].setText(self.labels[i])
                self.mainLayout.addWidget(self.ckb[i], il, 0, 1, 2)
                self.ckb[i].stateChanged.connect(self.checked)
                if values[i]:
                    self.ckb[i].setChecked(True)
                    self.ckState[i] = True
                    self.ck_order[i] = max(self.ck_order) + 1
                    self.check = False
                ilin += 1
            elif types[i].lower() == 'b':
                self.dlabels.append(None)
                self.dlines.append(None)
                self.ckb.append(None)
                self.rbtn.append(None)
                self.btngroup.append(None)
                self.combo.append(QtWidgets.QComboBox())
                for il, lab in enumerate(self.labels[i]):
                    self.combo[i].addItem(lab)
                ilin += 1
                self.mainLayout.addWidget(self.combo[i], ilin, 0, 1, 1)
                il_add += 1
        ilin += 2
        il = ilin + il_add
        self.mainLayout.addWidget(self.YesBtn, il, 0)
        self.mainLayout.addWidget(self.CancelBtn, il, 1)
        self.YesBtn.setDefault(True)
        self.YesBtn.clicked.connect(self.on_YesButton_clicked)
        self.CancelBtn.clicked.connect(self.on_CancelButton_clicked)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
#        self.setModal(True)
        self.setWindowTitle(title)
        self.show()

    def radio_checked(self, button):
        self.radio = True
        self.checked_field = int(button.property("groupName"))
        self.on_YesButton_clicked()

    def checked(self, checked):
        """
        Actions executed if a check box has changed its state.
        if a box has been checked, the function searches the one which was
        checked using self.ckState as indicator (this variable contains the
        state of all check boxes before the click) Its click-order is stored
        and self.ckState is changed. In addition, for check boxes, the
        explanation text is changed and the order of checking is indicated.
        If a box is unchecked, this is stored in self.ckState and the order
        numbers indicated are changed if necessary.

        Parameters
        ----------
        checked : QtWidget.QCheckBox.checkState
            state of a checkbox after clicking into it

        Returns
        -------
        None.

        """
        from PyQt5.QtCore import Qt
# If a check box is checked, search the one which has been checked and
# do necessary changes
# If self.ckb has None value, the corresponding entry is not a checkbox
        if checked == Qt.Checked:
            for i, ck in enumerate(self.ckb):
                if not ck:
                    continue
# If self.ckb.checkState is checked after click, set ckState to True and do
# changes
                if ck.checkState() == Qt.Checked:
                    self.ckState[i] = True
# If checkbox nr i was not checked, increase the number of checked boxes
#    (n_checked) and store the order of checkin in self.ck_order
                    if self.ck_order[i] == 0:
                        self.n_checked += 1
                        self.ck_order[i] = self.n_checked
# if checkboxis checked, change the label, indicating the order in which it was
#    checked.
                        self.ckb[i].setText(f"{self.labels[i]} "
                                            f"({self.ck_order[i]})")
                        if self.types[i] == "C":
                            self.check = True
                            self.checked_field = i
                            self.on_YesButton_clicked()
                        # break
# If self.ckb.checkState is still unchecked, set ckState to Falsee
                else:
                    self.ckState[i] = False
# If click has unchecked a checkbox, do necessary changes
# If self.ckb has None value, the corresponding entry is not a checkbox
        else:
            for i, ck in enumerate(self.ckb):
                if not ck:
                    continue
# If self.ckb.checkState is still checked, set ckState to True
                if ck.checkState() == Qt.Checked:
                    self.ckState[i] = True
# If checkbox is no longer checked but it was (self.ckState), the unchecked box
#    is found
                else:
                    if self.ckState[i] is True:
                        self.ckState[i] = False
                        n = self.ck_order[i]
# reset ck_order to 0 (indicating also unchecked box)
                        self.ck_order[i] = 0
# Reset label to initial value (changes only for function "False_Color")
                        self.ckb[i].setText(self.labels[i])
# For all boxes that were checked later than the unchecked one, reduce their
#    checking order by 1
                        for j in range(len(self.ck_order)):
                            if self.ck_order[j] > n:
                                self.ck_order[j] -= 1
                                self.ckb[j].setText(f"{self.labels[j]} "
                                                    f"({self.ck_order[j]})")
                        self.n_checked -= 1
                        if self.types[i] == "C":
                            self.check = True
                            self.checked_field = i
                            self.on_YesButton_clicked()
                        # break
        self.show()

    def on_YesButton_clicked(self):
        from PyQt5.QtCore import Qt
        n_checked = 0
        for ck in self.ckb:
            if not ck:
                continue
            if ck.checkState() == Qt.Checked:
                n_checked += 1
        self.Dfinish = True
        self.Dbutton = True
        self.close()

    def on_CancelButton_clicked(self):
        self.Dfinish = True
        self.Dbutton = False
        self.close()


def dialog(labels, types, values, title="Title"):
    """
      Wrapper for class Dialog. Two buttons are shown for finishing: Ok and
      Cancel

      Parameters
      ----------
      labels : list of strings
          Explanatory text put beside the fields foe data entrance.
          If values==None, label is interpreted as information/Warning text
          For a series of radiobuttons or Comboboxes, all corresponding labels
          are given as a list within the list.
      types : list of strings (length as labels).
          Possible values:
              "c": to define a check box
              "e": to define an editable box (lineEdit)
              "l": To define a simple label, comment text
              'r': to define a series of radiobuttons (the corresponding list
                   of labels is considered as one single label for the
                   numbering of labels, types and values)
              "b": to define a combobox (dropdown menu)
        values (list, string, float or int): initial values for LineEdit
            - Initial values for LineEdit fields (float, int or str)
            - None for comboboxes
            - >0 for check box if it should be checked from the beginning,
              0 else.
            - Number of radiobutton to be activated by default (natural
              numbering, starting at 1, not at 0).
            - For labels, it may be "b" (bold text), "i" (italic) or anything
              else, including None, for standard text.
      title : str, default: "Title"
          Title of the dialog box.

      Returns
      -------
      results : list of bool
          Response of each data entrance field. Should be transformed to the
          needed data format (int, float...) in the calling function
          For radiobuttons, the returned value indicates the number of the
          active button (counting starts at 0). For checkboxes, the returned
          value is -1 if the box was not checked and the position at which
          the box was checked (starting at 0) if it has been checked
      Dbutton: bool
          If True, "Apply" button has been pressed to finish dialog, if False
          "Cancel" button has been pressed.

    """
    from PyQt5.QtWidgets import QApplication
    from PyQt5 import QtCore
    import sys
    _ = QApplication(sys.argv)
    while True:
        c_flag = -1
        r_flag = -1
        lab = []
        typ = []
        val = []
        field = []
        radio = []
        check = []
        for i, t in enumerate(types):
            icolon = 0
            if t == "C":
                lab.append(labels[i])
                typ.append(t)
                val.append(values[i])
                field.append(i)
                radio.append(None)
                check.append(values[i])
                c_flag = len(lab)-1
                r_flag = -1
            elif t == "R":
                lab.append(labels[i])
                typ.append(t)
                val.append(values[i])
                field.append(i)
                radio.append(values[i])
                check.append(None)
                r_flag = len(lab)-1
                c_flag = -1
            else:
                if r_flag > -1 and labels[i][0] == "@":
                    icolon = labels[i].index(":")
                    if icolon < 2:
                        print(f"Error: Field {i}: Colon missing")
                        _ = QtWidgets.QMessageBox.warning(
                            None, "Warning", "the radio button indicator is "
                            f"capitalized but field {i} does not start with @."
                            "\n\nCapitalization is removed.",
                            QtWidgets.QMessageBox.Close,
                            QtWidgets.QMessageBox.Close)
                        r_flag = -1
                        lab.append(labels[i])
                        typ.append(types[i])
                        val.append(values[i])
                        field.append(i)
                        radio.append(None)
                        check.append(None)
                        continue
                    index = int(labels[i][1:icolon])
                    icolon += 1
                    if val[r_flag] == index:
                        lab.append(labels[i][icolon:])
                        typ.append(types[i])
                        val.append(values[i])
                        field.append(i)
                        radio.append(None)
                        check.append(None)
                    continue
                if c_flag > -1 and labels[i][0] == "@":
                    icolon = labels[i].index(":")
                    if icolon < 2:
                        print(f"Error: Field {i}: Colon missing")
                        _ = QtWidgets.QMessageBox.warning(
                            None, "Warning", "The check box indicator is "
                            f"capitalized but field {i} does not start with @."
                            "\n\nCapitalization is removed.",
                            QtWidgets.QMessageBox.Close,
                            QtWidgets.QMessageBox.Close)
                        r_flag = -1
                        lab.append(labels[i])
                        typ.append(types[i])
                        val.append(values[i])
                        field.append(i)
                        radio.append(None)
                        check.append(None)
                        continue
                    index = int(labels[i][1:icolon])
                    if val[c_flag] > 0:
                        vv = 1
                    else:
                        vv = 0
                    icolon += 1
                    if vv == index:
                        lab.append(labels[i][icolon:])
                        typ.append(types[i])
                        val.append(values[i])
                        field.append(i)
                        radio.append(None)
                        check.append(None)
                    continue
                lab.append(labels[i][icolon:])
                typ.append(types[i])
                val.append(values[i])
                field.append(i)
                radio.append(None)
                check.append(None)
                c_flag = -1
                r_flag = -1
        D = Dialog(lab, typ, val, title)
        D.Dfinish = False
        while (D.Dfinish is not True):
            QtCore.QCoreApplication.processEvents()

        results = []
        for i in range(len(labels)):
            results.append(None)
        iline = 0
        lv = len(values)
        if lv > 0:
            for it, t in enumerate(typ):
                if t.lower() == "e":
                    results[field[it]] = D.dlines[iline].text()
                    values[field[it]] = D.dlines[iline].text()
                    iline += 1
                elif t.lower() == "r":
                    for i in range(len(lab[it])):
                        iline += 1
                        if D.rbtn[it][i].isChecked():
                            results[field[it]] = i
                            values[field[it]] = i
                elif t.lower() == 'c':
                    results[field[it]] = D.ck_order[it]-1
                    if results[field[it]] < 0:
                        values[field[it]] = 0
                    else:
                        values[field[it]] = 1
                    iline += 1
                elif t.lower() == "b":
                    results[field[it]] = D.combo[it].currentIndex()
                    iline += 1
                else:
                    iline += 1
        if D.radio is False and D.check is False:
            break
        it = D.checked_field
        if D.radio:
            D.radio = False
            values[field[it]] = results[field[it]]+1
        if D.check:
            D.check = False
            if results[field[it]] < 0:
                values[field[it]] = 0
            else:
                values[field[it]] = results[field[it]]+1
    return results, D.Dbutton
