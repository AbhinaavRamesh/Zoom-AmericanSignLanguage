import Landing from "../index";
import { shallow } from "enzyme";
import React from 'react';
describe("Landing", () => {
  it("should render correctly", () => {
    let LandingSnapshot = shallow(<Landing/>);
    expect(LandingSnapshot).toMatchSnapshot();
  });
});